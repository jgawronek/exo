from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import anyio
from anyio import BrokenResourceError, ClosedResourceError, current_time, to_thread
from loguru import logger

from exo.download.download_utils import (
    InsufficientDiskSpaceError,
    RepoDownloadProgress,
    copy_shared_model_to_local,
    delete_model,
    is_read_only_model_dir,
    is_shared_model_path,
    map_repo_download_progress_to_download_progress_data,
    measure_model_directory,
    resolve_existing_model,
    select_download_dir,
    writable_model_dirs,
)
from exo.download.shard_downloader import ShardDownloader
from exo.download.shared_models_dir import is_shared_copy_source
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
)
from exo.shared.constants import EXO_DEFAULT_MODELS_DIR, EXO_MODELS_READ_ONLY_DIRS
from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelId
from exo.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    NodeDownloadProgress,
)
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
    DownloadProgressData,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender
from exo.utils.task_group import TaskGroup


@dataclass
class DownloadCoordinator:
    node_id: NodeId
    shard_downloader: ShardDownloader
    download_command_receiver: Receiver[ForwarderDownloadCommand]
    event_sender: Sender[Event]
    offline: bool = False

    # Local state
    download_status: dict[ModelId, DownloadProgress] = field(default_factory=dict)
    active_downloads: dict[ModelId, anyio.CancelScope] = field(default_factory=dict)

    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)
    _stopped: anyio.Event = field(init=False, default_factory=anyio.Event)

    # Per-model throttle for download progress events
    _last_progress_time: dict[ModelId, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.shard_downloader.on_progress(self._download_progress_callback)

    @staticmethod
    def _default_model_dir(model_id: ModelId) -> str:
        return str(EXO_DEFAULT_MODELS_DIR / model_id.normalize())

    def _completed_from_path(
        self,
        shard: ShardMetadata,
        found: Path,
        total: Memory,
    ) -> DownloadCompleted:
        return DownloadCompleted(
            shard_metadata=shard,
            node_id=self.node_id,
            total=total,
            model_directory=str(found),
            read_only=is_read_only_model_dir(found),
            on_share=is_shared_model_path(found),
        )

    async def _download_progress_callback(
        self, callback_shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        model_id = callback_shard.model_card.model_id
        throttle_interval_secs = 1.0

        try:
            if progress.status == "complete":
                found = await to_thread.run_sync(
                    resolve_existing_model, model_id, callback_shard.model_card
                )
                if found is not None:
                    completed = self._completed_from_path(
                        callback_shard, found, progress.total
                    )
                else:
                    completed = DownloadCompleted(
                        shard_metadata=callback_shard,
                        node_id=self.node_id,
                        total=progress.total,
                        model_directory=self._default_model_dir(model_id),
                    )
                self.download_status[model_id] = completed
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=completed)
                )
                self._last_progress_time.pop(model_id, None)
            elif (
                progress.status == "in_progress"
                and current_time() - self._last_progress_time.get(model_id, 0.0)
                > throttle_interval_secs
            ):
                ongoing = DownloadOngoing(
                    node_id=self.node_id,
                    shard_metadata=callback_shard,
                    download_progress=map_repo_download_progress_to_download_progress_data(
                        progress
                    ),
                    model_directory=self._default_model_dir(model_id),
                )
                self.download_status[model_id] = ongoing
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=ongoing)
                )
                self._last_progress_time[model_id] = current_time()
        except (BrokenResourceError, ClosedResourceError):
            logger.debug(
                f"Event stream closed while sending download progress for {model_id}, skipping update"
            )

    async def run(self) -> None:
        logger.info(
            f"Starting DownloadCoordinator{' (offline mode)' if self.offline else ''}"
        )
        try:
            async with self._tg as tg:
                tg.start_soon(self._command_processor)
                tg.start_soon(self._emit_existing_download_progress)
        except* (EventRouterBrokenResourceError, EventRouterClosedResourceError):
            # Event router has been closed (try-star syntax handles error groups)
            pass
        finally:
            # don't forget to clean up resources
            self.download_command_receiver.close()
            self.event_sender.close()

            self._stopped.set()

    async def shutdown(self) -> None:
        self._tg.cancel_tasks()
        await self._stopped.wait()

    async def _command_processor(self) -> None:
        with self.download_command_receiver as commands:
            async for cmd in commands:
                # Only process commands targeting this node
                if cmd.command.target_node_id != self.node_id:
                    continue

                match cmd.command:
                    case StartDownload(shard_metadata=shard):
                        await self._start_download(shard)
                    case DeleteDownload(model_id=model_id):
                        await self._delete_download(model_id)
                    case CancelDownload(model_id=model_id):
                        await self._cancel_download(model_id)

    async def _cancel_download(self, model_id: ModelId) -> None:
        if model_id in self.active_downloads and model_id in self.download_status:
            logger.info(f"Cancelling download for {model_id}")
            self.active_downloads[model_id].cancel()
            current_status = self.download_status[model_id]
            downloaded = Memory()
            total = Memory()
            if isinstance(current_status, DownloadOngoing):
                downloaded = current_status.download_progress.downloaded
                total = current_status.download_progress.total
            pending = DownloadPending(
                shard_metadata=current_status.shard_metadata,
                node_id=self.node_id,
                model_directory=self._default_model_dir(model_id),
                downloaded=downloaded,
                total=total,
            )
            self.download_status[model_id] = pending
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )

    async def _start_download(self, shard: ShardMetadata) -> None:
        model_id = shard.model_card.model_id

        # Check if already downloading, complete, or recently failed. A model
        # "complete" on a copy-to-local share is the one exception: that
        # completion advertises the copy *source*, and the download it asks
        # for is the copy onto local disk.
        if model_id in self.download_status:
            status = self.download_status[model_id]
            if isinstance(status, (DownloadOngoing, DownloadFailed)) or (
                isinstance(status, DownloadCompleted)
                and not is_shared_copy_source(status.model_directory)
            ):
                logger.debug(
                    f"Download for {model_id} already in progress, complete, or failed, skipping"
                )
                return

        # Check all model directories for pre-existing complete models
        found_path = await to_thread.run_sync(
            resolve_existing_model, model_id, shard.model_card
        )
        if found_path is not None:
            if is_shared_copy_source(found_path):
                logger.info(
                    f"DownloadCoordinator: Copying {model_id} from shared storage at {found_path}"
                )
                await self._start_copy_from_share(shard, found_path)
                return
            logger.info(f"DownloadCoordinator: Model {model_id} found at {found_path}")
            completed = self._completed_from_path(
                shard, found_path, shard.model_card.storage_size
            )
            self.download_status[model_id] = completed
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=completed)
            )
            return

        # Emit pending status
        progress = DownloadPending(
            shard_metadata=shard,
            node_id=self.node_id,
            model_directory=self._default_model_dir(model_id),
        )
        self.download_status[model_id] = progress
        await self.event_sender.send(NodeDownloadProgress(download_progress=progress))

        # Check initial status from downloader
        initial_progress = (
            await self.shard_downloader.get_shard_download_status_for_shard(shard)
        )

        if initial_progress.status == "complete":
            found = await to_thread.run_sync(
                resolve_existing_model, model_id, shard.model_card
            )
            if found is not None:
                completed = self._completed_from_path(
                    shard, found, initial_progress.total
                )
            else:
                completed = DownloadCompleted(
                    shard_metadata=shard,
                    node_id=self.node_id,
                    total=initial_progress.total,
                    model_directory=self._default_model_dir(model_id),
                )
            self.download_status[model_id] = completed
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=completed)
            )
            return

        if self.offline:
            logger.warning(
                f"Offline mode: model {model_id} is not fully available locally, cannot download"
            )
            failed = DownloadFailed(
                shard_metadata=shard,
                node_id=self.node_id,
                error_message=f"Model files not found locally in offline mode: {model_id}",
                model_directory=self._default_model_dir(model_id),
            )
            self.download_status[model_id] = failed
            await self.event_sender.send(NodeDownloadProgress(download_progress=failed))
            return

        # Start actual download
        self._start_download_task(shard, initial_progress)

    def _start_download_task(
        self, shard: ShardMetadata, initial_progress: RepoDownloadProgress
    ) -> None:
        model_id = shard.model_card.model_id

        # Emit ongoing status
        status = DownloadOngoing(
            node_id=self.node_id,
            shard_metadata=shard,
            download_progress=map_repo_download_progress_to_download_progress_data(
                initial_progress
            ),
            model_directory=self._default_model_dir(model_id),
        )
        self.download_status[model_id] = status
        self.event_sender.send_nowait(NodeDownloadProgress(download_progress=status))

        async def download_wrapper(cancel_scope: anyio.CancelScope) -> None:
            try:
                with cancel_scope:
                    await self.shard_downloader.ensure_shard(shard)
            except Exception as e:
                logger.error(f"Download failed for {model_id}: {e}")
                failed = DownloadFailed(
                    shard_metadata=shard,
                    node_id=self.node_id,
                    error_message=str(e),
                    model_directory=self._default_model_dir(model_id),
                )
                self.download_status[model_id] = failed
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=failed)
                )
            except anyio.get_cancelled_exc_class():
                # ignore cancellation - let cleanup do its thing
                pass
            finally:
                self.active_downloads.pop(model_id, None)

        scope = anyio.CancelScope()
        self._tg.start_soon(download_wrapper, scope)
        self.active_downloads[model_id] = scope

    async def _start_copy_from_share(
        self, shard: ShardMetadata, source_dir: Path
    ) -> None:
        """Copy a model off the copy-to-local share onto this node's disk.

        Runs as a download in every observable way — ongoing progress, a
        completion carrying the local directory, cancellation through
        ``active_downloads`` — so placement and the dashboard need no special
        case for it.
        """
        model_id = shard.model_card.model_id
        normalized = model_id.normalize()
        total = await measure_model_directory(source_dir)

        def pick_target_root() -> Path | InsufficientDiskSpaceError:
            # Resume into whichever writable directory already holds a partial
            # copy; otherwise pick by free space like any other download.
            for candidate_dir in writable_model_dirs():
                if (candidate_dir / normalized).is_dir():
                    return candidate_dir
            try:
                return select_download_dir(total.in_bytes)
            except InsufficientDiskSpaceError as error:
                return error

        target_root = await to_thread.run_sync(pick_target_root)
        if isinstance(target_root, InsufficientDiskSpaceError):
            failed = DownloadFailed(
                shard_metadata=shard,
                node_id=self.node_id,
                error_message=(
                    f"Not enough local disk space to copy {model_id} from "
                    f"shared storage: {target_root}. Free up space or turn "
                    "off 'copy to local disk' on the share."
                ),
                model_directory=self._default_model_dir(model_id),
            )
            self.download_status[model_id] = failed
            await self.event_sender.send(NodeDownloadProgress(download_progress=failed))
            return
        target_dir = target_root / normalized

        started_at = current_time()
        resumed_bytes: int | None = None

        async def copy_progress(
            copied: Memory, copy_total: Memory, completed_files: int, total_files: int
        ) -> None:
            nonlocal resumed_bytes
            # The first report carries whatever an earlier attempt already
            # copied; speed must count this session's bytes only.
            if resumed_bytes is None:
                resumed_bytes = copied.in_bytes
            now = current_time()
            if now - self._last_progress_time.get(model_id, 0.0) <= 1.0:
                return
            this_session = copied.in_bytes - resumed_bytes
            speed = this_session / max(now - started_at, 0.001)
            remaining = copy_total.in_bytes - copied.in_bytes
            ongoing = DownloadOngoing(
                node_id=self.node_id,
                shard_metadata=shard,
                download_progress=DownloadProgressData(
                    total=copy_total,
                    downloaded=copied,
                    downloaded_this_session=Memory.from_bytes(this_session),
                    completed_files=completed_files,
                    total_files=total_files,
                    speed=speed,
                    eta_ms=int(remaining / speed * 1000) if speed > 0 else 0,
                    files={},
                ),
                model_directory=str(target_dir),
                from_share=True,
            )
            self.download_status[model_id] = ongoing
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=ongoing)
            )
            self._last_progress_time[model_id] = now

        async def copy_wrapper(cancel_scope: anyio.CancelScope) -> None:
            try:
                with cancel_scope:
                    await copy_shared_model_to_local(
                        source_dir, target_dir, copy_progress
                    )
                    completed = DownloadCompleted(
                        shard_metadata=shard,
                        node_id=self.node_id,
                        total=total,
                        model_directory=str(target_dir),
                    )
                    self.download_status[model_id] = completed
                    await self.event_sender.send(
                        NodeDownloadProgress(download_progress=completed)
                    )
            except Exception as copy_error:
                logger.error(f"Copy from share failed for {model_id}: {copy_error}")
                failed = DownloadFailed(
                    shard_metadata=shard,
                    node_id=self.node_id,
                    error_message=(f"Copying from shared storage failed: {copy_error}"),
                    model_directory=self._default_model_dir(model_id),
                )
                self.download_status[model_id] = failed
                await self.event_sender.send(
                    NodeDownloadProgress(download_progress=failed)
                )
            except anyio.get_cancelled_exc_class():
                # ignore cancellation - let cleanup do its thing
                pass
            finally:
                self.active_downloads.pop(model_id, None)
                self._last_progress_time.pop(model_id, None)

        scope = anyio.CancelScope()
        self._tg.start_soon(copy_wrapper, scope)
        self.active_downloads[model_id] = scope

    async def _delete_download(self, model_id: ModelId) -> None:
        # Protect read-only models from deletion
        if model_id in self.download_status:
            current = self.download_status[model_id]
            if isinstance(current, DownloadCompleted) and current.read_only:
                logger.warning(f"Refusing to delete read-only model {model_id}")
                return

        # Cancel if active
        if model_id in self.active_downloads:
            logger.info(f"Cancelling active download for {model_id} before deletion")
            self.active_downloads[model_id].cancel()

        # Delete from disk
        logger.info(f"Deleting model files for {model_id}")
        deleted = await delete_model(model_id)

        if deleted:
            logger.info(f"Successfully deleted model {model_id}")
        else:
            logger.warning(f"Model {model_id} was not found on disk")

        # Emit pending status to reset UI state, then remove from local tracking
        if model_id in self.download_status:
            current_status = self.download_status[model_id]
            pending = DownloadPending(
                shard_metadata=current_status.shard_metadata,
                node_id=self.node_id,
                model_directory=self._default_model_dir(model_id),
            )
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=pending)
            )
            del self.download_status[model_id]

    async def _emit_existing_download_progress(self) -> None:
        while True:
            try:
                logger.debug(
                    "DownloadCoordinator: Fetching and emitting existing download progress..."
                )
                async for (
                    _,
                    progress,
                ) in self.shard_downloader.get_shard_download_status():
                    model_id = progress.shard.model_card.model_id

                    # Active downloads emit progress via the callback — don't overwrite
                    if model_id in self.active_downloads:
                        continue

                    if progress.status == "complete":
                        found = await to_thread.run_sync(
                            resolve_existing_model,
                            model_id,
                            progress.shard.model_card,
                        )
                        if found is not None:
                            # A model that is only complete on a copy-to-local
                            # share is advertised as available from it — but
                            # never over a failed copy attempt, or the failure
                            # would flap back to "complete" and re-trigger a
                            # copy that keeps failing.
                            if is_shared_copy_source(found) and isinstance(
                                self.download_status.get(model_id), DownloadFailed
                            ):
                                continue
                            status: DownloadProgress = self._completed_from_path(
                                progress.shard, found, progress.total
                            )
                        else:
                            status = DownloadCompleted(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                total=progress.total,
                                model_directory=self._default_model_dir(model_id),
                            )
                    elif progress.status in ["in_progress", "not_started"]:
                        # TODO(ciaran): temporary solution
                        # Don't downgrade a model that is already confirmed complete.
                        # A failure must survive too: this scan derives status
                        # purely from bytes on disk, so it would rewrite a
                        # DownloadFailed as DownloadPending and destroy the
                        # error_message with it — leaving the dashboard showing
                        # a bare "WAITING" with no reason. The failure stays
                        # until the next attempt reports a new outcome.
                        if isinstance(
                            self.download_status.get(model_id),
                            (DownloadCompleted, DownloadFailed),
                        ):
                            continue
                        # The per-file size check compares local files against
                        # the latest HF "main" revision, which is a moving
                        # target.  When HF updates text files (README, YAML,
                        # jinja) in a new commit, the cached file list has new
                        # sizes while local files still match the old revision.
                        # Fall back to the authoritative completeness check
                        # (is_model_directory_complete) which validates that all
                        # safetensors weight files are present.
                        found = await to_thread.run_sync(
                            resolve_existing_model,
                            model_id,
                            progress.shard.model_card,
                        )
                        if found is not None:
                            status = self._completed_from_path(
                                progress.shard, found, progress.total
                            )
                        elif progress.downloaded_this_session.in_bytes == 0:
                            if progress.downloaded.in_bytes == 0:
                                # Nothing local and nothing happening: absence
                                # in state carries the same information, and a
                                # pending record per catalog model per node
                                # bloats every state poll.
                                continue
                            status = DownloadPending(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                model_directory=self._default_model_dir(model_id),
                                downloaded=progress.downloaded,
                                total=progress.total,
                            )
                        else:
                            status = DownloadOngoing(
                                node_id=self.node_id,
                                shard_metadata=progress.shard,
                                download_progress=map_repo_download_progress_to_download_progress_data(
                                    progress
                                ),
                                model_directory=self._default_model_dir(model_id),
                            )
                    else:
                        continue

                    self.download_status[progress.shard.model_card.model_id] = status
                    await self.event_sender.send(
                        NodeDownloadProgress(download_progress=status)
                    )
                # Scan read-only directories for pre-downloaded models
                if EXO_MODELS_READ_ONLY_DIRS:
                    for card in await model_cards.card_cache.list_all():
                        mid = card.model_id
                        if mid in self.active_downloads:
                            continue
                        if isinstance(
                            self.download_status.get(mid),
                            (DownloadCompleted, DownloadOngoing, DownloadFailed),
                        ):
                            continue
                        found = await to_thread.run_sync(
                            resolve_existing_model, mid, card
                        )
                        if found is not None and is_read_only_model_dir(found):
                            path_shard = PipelineShardMetadata(
                                model_card=card,
                                device_rank=0,
                                world_size=1,
                                start_layer=0,
                                end_layer=card.n_layers,
                                n_layers=card.n_layers,
                            )
                            path_completed: DownloadProgress = (
                                self._completed_from_path(
                                    path_shard, found, card.storage_size
                                )
                            )
                            self.download_status[mid] = path_completed
                            await self.event_sender.send(
                                NodeDownloadProgress(download_progress=path_completed)
                            )

                logger.debug(
                    "DownloadCoordinator: Done emitting existing download progress."
                )
            except Exception as e:
                logger.error(
                    f"DownloadCoordinator: Error emitting existing download progress: {e}"
                )
            await anyio.sleep(60)
