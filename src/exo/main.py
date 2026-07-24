import argparse
import multiprocessing as mp
import os
import resource
import signal
import sys
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Self

import anyio
from anyio import CancelScope, to_thread
from daemon import DaemonContext  # pyright: ignore[reportMissingTypeStubs]
from exo_rs import Pidfile, PidfileError
from loguru import logger
from pydantic import PositiveInt

import exo.routing.topics as topics
from exo import __version__
from exo.api.main import API
from exo.download.coordinator import DownloadCoordinator
from exo.download.impl_shard_downloader import exo_shard_downloader
from exo.master.main import Master
from exo.routing.event_router import EventRouter, SnapshotTransport
from exo.routing.router import Router, get_node_zid
from exo.shared.constants import (
    EXO_DEFAULT_MODELS_DIR,
    EXO_EVENT_LOG_DIR,
    EXO_LOG,
    EXO_PID_FILE,
)
from exo.shared.election import Election, ElectionResult
from exo.shared.environment import get_compatible_environment_value
from exo.shared.logging import logger_cleanup, logger_setup
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.state import State
from exo.utils import STDIO_FDS
from exo.utils.channels import Receiver, channel
from exo.utils.pydantic_ext import FrozenModel
from exo.utils.replica_store import ReplicaStore, ReplicaStoreError
from exo.utils.state_replica import StateReplica
from exo.utils.task_group import TaskGroup
from exo.worker.main import Worker


def _create_snapshot_transport(router: Router) -> SnapshotTransport:
    return SnapshotTransport(
        request_sender=router.sender(topics.STATE_SNAPSHOT_REQUESTS),
        request_receiver=router.receiver(topics.STATE_SNAPSHOT_REQUESTS),
        manifest_sender=router.sender(topics.STATE_SNAPSHOT_MANIFESTS),
        manifest_receiver=router.receiver(topics.STATE_SNAPSHOT_MANIFESTS),
        chunk_sender=router.sender(topics.STATE_SNAPSHOT_CHUNKS),
        chunk_receiver=router.receiver(topics.STATE_SNAPSHOT_CHUNKS),
        unavailable_sender=router.sender(topics.STATE_SNAPSHOT_UNAVAILABLE),
        unavailable_receiver=router.receiver(topics.STATE_SNAPSHOT_UNAVAILABLE),
    )


@dataclass
class Node:
    router: Router
    event_router: EventRouter
    download_coordinator: DownloadCoordinator | None
    worker: Worker | None
    election: Election  # Every node participates in election, as we do want a node to become master even if it isn't a master candidate if no master candidates are present.
    election_result_receiver: Receiver[ElectionResult]
    master: Master | None
    api: API | None
    state_replica: StateReplica
    replica_store: ReplicaStore

    node_id: NodeId
    offline: bool
    _api_port: int
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)
    _has_completed_election: bool = field(init=False, default=False)
    _recovery_cancel_scope: CancelScope | None = field(init=False, default=None)

    @classmethod
    async def create(cls, args: "Args") -> Self:
        node_id = get_node_zid()
        session_id = SessionId(master_node_id=node_id, election_clock=0)
        router = Router.create(
            node_id,
            namespace=args.namespace,
            listen_port=args.zenoh_port,
            discovery_service_port=args.discovery_port,
        )
        await router.register_topic(topics.GLOBAL_EVENTS)
        await router.register_topic(topics.LOCAL_EVENTS)
        await router.register_topic(topics.COMMANDS)
        await router.register_topic(topics.ELECTION_MESSAGES)
        await router.register_topic(topics.CONNECTION_MESSAGES)
        await router.register_topic(topics.DOWNLOAD_COMMANDS)
        await router.register_topic(topics.STATE_SNAPSHOT_REQUESTS)
        await router.register_topic(topics.STATE_SNAPSHOT_MANIFESTS)
        await router.register_topic(topics.STATE_SNAPSHOT_CHUNKS)
        await router.register_topic(topics.STATE_SNAPSHOT_UNAVAILABLE)
        replica_store = ReplicaStore(EXO_EVENT_LOG_DIR / "replica")
        try:
            recovered_replica = replica_store.recover()
        except ReplicaStoreError as exception:
            logger.opt(exception=exception).warning(
                "Ignoring invalid local state replica; recovering from the cluster"
            )
            replica_store.clear()
            recovered_replica = None
        if recovered_replica is None:
            state_replica = StateReplica(
                session=session_id,
                initial_state=State(),
                ready=True,
            )
        else:
            state_replica = recovered_replica
            state_replica.rebase_session(session_id)
            replica_store.write_checkpoint(
                session_id,
                state_replica.state,
                compact_journal=True,
            )
        event_router = EventRouter(
            session_id,
            command_sender=router.sender(topics.COMMANDS),
            external_outbound=router.sender(topics.LOCAL_EVENTS),
            external_inbound=router.receiver(topics.GLOBAL_EVENTS),
            node_id=node_id,
            state_replica=state_replica,
            snapshot_transport=_create_snapshot_transport(router),
            replica_store=replica_store,
        )

        logger.info(f"Starting node {node_id}")

        # Errors the very first time exo is run as dir doesn't exist
        EXO_DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Create DownloadCoordinator (unless --no-downloads)
        if not args.no_downloads:
            download_coordinator = DownloadCoordinator(
                node_id,
                exo_shard_downloader(offline=args.offline),
                event_sender=event_router.sender(),
                download_command_receiver=router.receiver(topics.DOWNLOAD_COMMANDS),
                offline=args.offline,
            )
        else:
            download_coordinator = None

        if args.spawn_api:
            api = API(
                node_id,
                port=args.api_port,
                event_receiver=event_router.receiver(),
                command_sender=router.sender(topics.COMMANDS),
                download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
                election_receiver=router.receiver(topics.ELECTION_MESSAGES),
                state_replica=state_replica,
            )
        else:
            api = None

        if not args.no_worker:
            worker = Worker(
                node_id,
                event_receiver=event_router.receiver(),
                event_sender=event_router.sender(),
                command_sender=router.sender(topics.COMMANDS),
                download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
                api_port=args.api_port,
                state_replica=state_replica,
            )
        else:
            worker = None

        # We start every node with a master
        master = Master(
            node_id,
            session_id,
            event_sender=event_router.sender(),
            global_event_sender=router.sender(topics.GLOBAL_EVENTS),
            local_event_receiver=router.receiver(topics.LOCAL_EVENTS),
            command_receiver=router.receiver(topics.COMMANDS),
            download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
            state_replica=state_replica,
            replica_store=replica_store,
        )

        er_send, er_recv = channel[ElectionResult]()
        election = Election(
            node_id,
            # If someone manages to assemble 1 MILLION devices into an exo cluster then. well done. good job champ.
            seniority=1_000_000 if args.force_master else 0,
            # nb: this DOES feedback right now. i have thoughts on how to address this,
            # but ultimately it seems not worth the complexity
            election_message_sender=router.sender(topics.ELECTION_MESSAGES),
            election_message_receiver=router.receiver(topics.ELECTION_MESSAGES),
            connection_message_receiver=router.receiver(topics.CONNECTION_MESSAGES),
            command_receiver=router.receiver(topics.COMMANDS),
            election_result_sender=er_send,
        )

        return cls(
            router,
            event_router,
            download_coordinator,
            worker,
            election,
            er_recv,
            master,
            api,
            state_replica,
            replica_store,
            node_id,
            args.offline,
            args.api_port,
        )

    async def run(self):
        async with self._tg as tg:
            signal.signal(signal.SIGINT, lambda _, __: self.shutdown())
            signal.signal(signal.SIGTERM, lambda _, __: self.shutdown())
            tg.start_soon(self.router.run)
            tg.start_soon(self.event_router.run)
            tg.start_soon(self.election.run)
            if self.download_coordinator:
                tg.start_soon(self.download_coordinator.run)
            if self.worker:
                tg.start_soon(self.worker.run)
            if self.master:
                tg.start_soon(self.master.run)
            if self.api:
                tg.start_soon(self.api.run)
            tg.start_soon(self._elect_loop)

    def shutdown(self):
        # if this is our second call to shutdown, just sys.exit
        if self._tg.cancel_called():
            import sys

            sys.exit(1)
        self._tg.cancel_tasks()

    async def _elect_loop(self):
        with self.election_result_receiver as results:
            async for result in results:
                # This function continues to have a lot of very specific entangled logic
                # At least it's somewhat contained

                # I don't like this duplication, but it's manageable for now.
                # TODO: This function needs refactoring generally

                # Ok:
                # On new master:
                # - Elect master locally if necessary
                # - Shutdown and re-create the worker
                # - Shut down and re-create the API

                if result.is_new_master:
                    if self._recovery_cancel_scope is not None:
                        self._recovery_cancel_scope.cancel()
                    self.event_router.shutdown()
                    if (
                        self.master is not None
                        and result.session_id.master_node_id != self.node_id
                    ):
                        await self.master.shutdown()
                        self.master = None
                    await self.event_router.wait_stopped()

                    preserve_state = result.session_id.master_node_id == self.node_id
                    if self.state_replica.ready and preserve_state:
                        self.state_replica.rebase_session(result.session_id)
                        try:
                            await to_thread.run_sync(
                                self.replica_store.write_compacted_checkpoint,
                                result.session_id,
                                self.state_replica.state,
                            )
                        except (OSError, ReplicaStoreError) as exception:
                            logger.opt(exception=exception).warning(
                                "Failed to persist rebased state replica"
                            )
                    else:
                        self.state_replica.start_unready_session(result.session_id)
                        self.replica_store.clear()
                    if result.session_id.master_node_id == self.node_id:
                        self.state_replica.mark_ready()

                    self.event_router = EventRouter(
                        result.session_id,
                        self.router.sender(topics.COMMANDS),
                        self.router.receiver(topics.GLOBAL_EVENTS),
                        self.router.sender(topics.LOCAL_EVENTS),
                        node_id=self.node_id,
                        state_replica=self.state_replica,
                        snapshot_transport=_create_snapshot_transport(self.router),
                        replica_store=self.replica_store,
                        allow_legacy_fallback=not self._has_completed_election,
                    )

                if (
                    result.session_id.master_node_id == self.node_id
                    and self.master is not None
                ):
                    assert not result.is_new_master, (
                        "cannot be new master if we remain master"
                    )
                    logger.info("Node elected Master - maintaining self")
                elif (
                    result.session_id.master_node_id == self.node_id
                    and self.master is None
                ):
                    logger.info("Node elected Master - promoting self")
                    self.master = Master(
                        self.node_id,
                        result.session_id,
                        event_sender=self.event_router.sender(),
                        global_event_sender=self.router.sender(topics.GLOBAL_EVENTS),
                        local_event_receiver=self.router.receiver(topics.LOCAL_EVENTS),
                        command_receiver=self.router.receiver(topics.COMMANDS),
                        download_command_sender=self.router.sender(
                            topics.DOWNLOAD_COMMANDS
                        ),
                        state_replica=self.state_replica,
                        replica_store=self.replica_store,
                    )
                    self._tg.start_soon(self.master.run)
                elif (
                    result.session_id.master_node_id != self.node_id
                    and self.master is not None
                ):
                    logger.info(
                        f"Node {result.session_id.master_node_id} elected master - demoting self"
                    )
                    await self.master.shutdown()
                    self.master = None
                else:
                    logger.info(
                        f"Node {result.session_id.master_node_id} elected master"
                    )
                if result.is_new_master:
                    self._tg.start_soon(self.event_router.run)
                    self._tg.start_soon(
                        self._activate_recovered_session,
                        result,
                        self.event_router,
                    )
                else:
                    if self.api and self.state_replica.ready:
                        self.api.unpause(result.won_clock)
                self._has_completed_election = True

    async def _activate_recovered_session(
        self,
        result: ElectionResult,
        event_router: EventRouter,
    ) -> None:
        with CancelScope() as recovery_scope:
            self._recovery_cancel_scope = recovery_scope
            try:
                await self.state_replica.wait_ready()
                if self.event_router is not event_router:
                    return
                if self.download_coordinator:
                    with anyio.fail_after(10):
                        await self.download_coordinator.shutdown()
                if self.worker:
                    with anyio.fail_after(10):
                        await self.worker.shutdown()
                if self.download_coordinator:
                    self.download_coordinator = DownloadCoordinator(
                        self.node_id,
                        exo_shard_downloader(offline=self.offline),
                        event_sender=event_router.sender(),
                        download_command_receiver=self.router.receiver(
                            topics.DOWNLOAD_COMMANDS
                        ),
                        offline=self.offline,
                    )
                    self._tg.start_soon(self.download_coordinator.run)
                if self.worker:
                    self.worker = Worker(
                        self.node_id,
                        event_receiver=event_router.receiver(),
                        event_sender=event_router.sender(),
                        command_sender=self.router.sender(topics.COMMANDS),
                        download_command_sender=self.router.sender(
                            topics.DOWNLOAD_COMMANDS
                        ),
                        api_port=self._api_port,
                        state_replica=self.state_replica,
                    )
                    self._tg.start_soon(self.worker.run)
                if self.api:
                    self.api.reset(result.won_clock, event_router.receiver())
            finally:
                if self._recovery_cancel_scope is recovery_scope:
                    self._recovery_cancel_scope = None


def main():
    # Parse args first => --help or bad args don't require PID-locking
    args = Args.parse()

    # Exit early if cannot acquire PID file
    try:
        pidfile = Pidfile(EXO_PID_FILE, 0o0600)
    except PidfileError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1) from e

    try:
        if args.legacy_daemon:
            # keep stdio backed by explicit /dev/null streams. multiprocessing spawn expects
            # valid stdio FDs; letting DaemonContext close/reopen them can break runner startup.
            for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
                if stream is not None:
                    stream.flush()
            stdin = open(os.devnull, "r")  # noqa: SIM115
            stdout = open(os.devnull, "w")  # noqa: SIM115
            stderr = open(os.devnull, "w")  # noqa: SIM115

            with DaemonContext(
                detach_process=True,
                files_preserve=[pidfile.as_raw_fd()],
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
            ):
                # cleanup loose file descriptors (as long as they aren't stdio)
                for f in (
                    f for f in (stdin, stdout, stderr) if f.fileno() not in STDIO_FDS
                ):
                    f.close()

                # 1) if daemonizing => fork then write PID
                try:
                    pidfile.write()
                except PidfileError as e:
                    print(e, file=sys.stderr)
                    raise SystemExit(1) from e
                main_inner(args)
        else:
            # 2) otherwise      => just write PID
            try:
                pidfile.write()
            except PidfileError as e:
                print(e, file=sys.stderr)
                raise SystemExit(1) from e
            main_inner(args)
    finally:
        pidfile.close()


def apply_runner_environment_overrides(
    args: "Args",
    environment: MutableMapping[str, str],
) -> None:
    if args.no_batch:
        environment["XEO_NO_BATCH"] = "1"
        environment["EXO_NO_BATCH"] = "1"

    if args.fast_synch is not None:
        fast_synch_value = "true" if args.fast_synch else "false"
        environment["XEO_FAST_SYNCH"] = fast_synch_value
        environment["EXO_FAST_SYNCH"] = fast_synch_value


def main_inner(args: "Args"):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, 65535), hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))

    mp.set_start_method("spawn", force=True)

    # TODO: Refactor the current verbosity system
    logger_setup(EXO_LOG, args.verbosity)

    logger.info(f"pid = {os.getpid()}")
    if get_compatible_environment_value(os.environ, "EXO_LIBP2P_NAMESPACE"):
        raise ValueError(
            "XEO_LIBP2P_NAMESPACE and EXO_LIBP2P_NAMESPACE have been removed; "
            "use XEO_ZENOH_NAMESPACE instead"
        )
    logger.info(
        "XEO_ZENOH_NAMESPACE: "
        f"{get_compatible_environment_value(os.environ, 'EXO_ZENOH_NAMESPACE')}"
    )

    if args.offline:
        logger.info("Running in OFFLINE mode — no internet checks, local models only")

    if args.bootstrap_peers:
        raise ValueError("Bootstrap peers has been temporarily removed")

    apply_runner_environment_overrides(args, os.environ)
    if args.no_batch:
        logger.info("Continuous batching disabled (--no-batch)")

    if args.fast_synch is True:
        logger.info("FAST_SYNCH forced ON")
    elif args.fast_synch is False:
        logger.info("FAST_SYNCH forced OFF")

    node = anyio.run(Node.create, args)
    try:
        anyio.run(node.run)
    except BaseException as exception:
        logger.opt(exception=exception).critical(
            "EXO terminated due to unhandled exception"
        )
        raise
    finally:
        logger.info("EXO Shutdown complete")
        logger_cleanup()


class Args(FrozenModel):
    verbosity: int = 0
    force_master: bool = False
    spawn_api: bool = False
    api_port: PositiveInt = 52415
    tb_only: bool = False
    no_worker: bool = False
    no_downloads: bool = False
    offline: bool = (
        get_compatible_environment_value(
            os.environ,
            "EXO_OFFLINE",
            "false",
        ).lower()
        == "true"
    )
    no_batch: bool = False
    fast_synch: bool | None = None  # None = auto, True = force on, False = force off
    legacy_daemon: bool = False
    bootstrap_peers: list[str] = []
    namespace: str
    zenoh_port: int
    discovery_port: int

    @classmethod
    def parse(cls) -> Self:
        parser = argparse.ArgumentParser(prog="XEO")
        default_verbosity = 0
        parser.add_argument(
            "-q",
            "--quiet",
            action="store_const",
            const=-1,
            dest="verbosity",
            default=default_verbosity,
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            dest="verbosity",
            default=default_verbosity,
        )
        parser.add_argument(
            "-m",
            "--force-master",
            action="store_true",
            dest="force_master",
        )
        parser.add_argument(
            "--no-api",
            action="store_false",
            dest="spawn_api",
        )
        parser.add_argument(
            "--api-port",
            type=int,
            dest="api_port",
            default=52415,
        )
        parser.add_argument(
            "--no-worker",
            action="store_true",
        )
        parser.add_argument(
            "--no-downloads",
            action="store_true",
            help="Disable the download coordinator (node won't download models)",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            default=get_compatible_environment_value(
                os.environ,
                "EXO_OFFLINE",
                "false",
            ).lower()
            == "true",
            help="Run in offline/air-gapped mode: skip internet checks, use only pre-staged local models",
        )
        parser.add_argument(
            "--no-batch",
            action="store_true",
            help="Disable continuous batching, use sequential generation",
        )
        parser.add_argument(
            "--legacy-daemon",
            action="store_true",
            help="Run as a legacy SysV-style background daemon using double-fork daemonization",
        )
        bootstrap_peers_environment_value = get_compatible_environment_value(
            os.environ,
            "EXO_BOOTSTRAP_PEERS",
        )
        parser.add_argument(
            "--bootstrap-peers",
            type=lambda s: [p for p in s.split(",") if p],
            default=bootstrap_peers_environment_value.split(",")
            if bootstrap_peers_environment_value
            else [],
            dest="bootstrap_peers",
            help="Comma-separated libp2p multiaddrs to dial on startup (env: XEO_BOOTSTRAP_PEERS)",
        )
        parser.add_argument(
            "--namespace",
            type=str,
            default=__version__,
            dest="namespace",
            help="Discovery namespace, nodes with different namespaces will not connect.",
        )
        parser.add_argument(
            "--zenoh-port",
            type=int,
            default=52414,
            dest="zenoh_port",
            help="Fixed TCP port for zenoh to listen.",
        )
        parser.add_argument(
            "--discovery-port",
            type=int,
            default=52413,
            dest="discovery_port",
            help="Fixed UDP port for the discovery service.",
        )
        fast_synch_group = parser.add_mutually_exclusive_group()
        fast_synch_group.add_argument(
            "--fast-synch",
            action="store_true",
            dest="fast_synch",
            default=None,
            help="Force MLX FAST_SYNCH on (for JACCL backend)",
        )
        fast_synch_group.add_argument(
            "--no-fast-synch",
            action="store_false",
            dest="fast_synch",
            help="Force MLX FAST_SYNCH off",
        )

        args = parser.parse_args()
        return cls(**vars(args))  # pyright: ignore[reportAny] - We are intentionally validating here, we can't do it statically
