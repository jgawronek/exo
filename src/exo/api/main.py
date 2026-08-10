import base64
import contextlib
import hashlib
import json
import os
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, cast
from uuid import uuid4

import anyio
from anyio import BrokenResourceError, ClosedResourceError, to_thread
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from hypercorn.utils import LifespanTimeoutError, ShutdownError
from loguru import logger

from exo.api.adapters.chat_completions import (
    chat_request_to_text_generation,
    collect_chat_response,
    generate_chat_stream,
)
from exo.api.adapters.claude import (
    claude_request_to_text_generation,
    collect_claude_response,
    generate_claude_stream,
)
from exo.api.adapters.ollama import (
    collect_ollama_chat_response,
    collect_ollama_generate_response,
    generate_ollama_chat_stream,
    generate_ollama_generate_stream,
    ollama_generate_request_to_text_generation,
    ollama_request_to_text_generation,
)
from exo.api.adapters.responses import (
    collect_responses_response,
    generate_responses_stream,
    responses_request_to_text_generation,
)
from exo.api.keepalive import with_sse_keepalive
from exo.api.types import (
    AddCustomModelParams,
    AdvancedImageParams,
    AwaitInstanceReadyMessage,
    AwaitInstanceTimeoutMessage,
    BenchChatCompletionRequest,
    BenchChatCompletionResponse,
    BenchImageGenerationResponse,
    BenchImageGenerationTaskParams,
    CancelCommandResponse,
    CancelDownloadParams,
    CancelDownloadResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CreateInstanceParams,
    CreateInstanceResponse,
    DeleteDownloadResponse,
    DeleteInstanceResponse,
    DeleteTracesRequest,
    DeleteTracesResponse,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    GenerationStats,
    HuggingFaceSearchResult,
    HuggingFaceTokenResponse,
    ImageData,
    ImageEditsTaskParams,
    ImageGenerationResponse,
    ImageGenerationStats,
    ImageGenerationTaskParams,
    ImageListItem,
    ImageListResponse,
    ImageSize,
    InstanceLinkBody,
    InstanceLinkResponse,
    ModelList,
    ModelListModel,
    ModelsStorageBrowseEntry,
    ModelsStorageBrowseResponse,
    ModelsStorageNetworkResponse,
    ModelsStorageNetworkServer,
    ModelsStorageNetworkShare,
    ModelsStorageNetworkVolume,
    ModelsStorageNodeStatus,
    ModelsStorageResponse,
    ModelsStorageShare,
    MountShareParams,
    MountShareResponse,
    PlaceInstanceParams,
    PlacementPreview,
    PlacementPreviewResponse,
    RebalanceInstanceResponse,
    SetHuggingFaceTokenParams,
    SetHuggingFaceTokenResponse,
    SetModelsStorageParams,
    SetModelsStorageResponse,
    SetModelsStorageShareParams,
    SetModelsStorageShareResponse,
    StartDownloadParams,
    StartDownloadResponse,
    ToolCall,
    TraceCategoryStats,
    TraceEventResponse,
    TraceListItem,
    TraceListResponse,
    TraceRankStats,
    TraceResponse,
    TraceStatsResponse,
    normalize_image_size,
)
from exo.api.types.claude_api import (
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
)
from exo.api.types.ollama_api import (
    OllamaCapability,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaModelDetails,
    OllamaModelTag,
    OllamaPsModel,
    OllamaPsResponse,
    OllamaShowRequest,
    OllamaShowResponse,
    OllamaTagsResponse,
)
from exo.api.types.openai_responses import (
    ResponsesRequest,
    ResponsesResponse,
)
from exo.download.download_utils import create_http_session
from exo.download.huggingface_utils import (
    delete_hf_token,
    get_hf_endpoint,
    get_hf_token,
    get_hf_token_source,
    mask_hf_token,
    set_hf_token,
)
from exo.download.share_mounts import (
    ShareMountError,
    discover_smb_servers,
    ensure_share_mounted,
    existing_nfs_mount_point,
    list_lan_shares,
)
from exo.download.shared_models_dir import (
    browse_shared_models_directories,
    get_shared_models_dir,
    list_network_volumes,
    resolve_shared_models_path,
)
from exo.master.image_store import ImageStore
from exo.master.placement import place_instance as get_instance_placements
from exo.master.placement_utils import (
    allocate_layers_by_expert_activity,
    allocate_layers_by_measured_speed,
    node_memory_with_pending_shutdowns,
    plan_pipeline_layer_shift_steps,
    projected_decode_compute_ms,
    validate_live_rebalance_steps,
)
from exo.routing.event_router import ReplicatedEventDelivery
from exo.shared.apply import apply
from exo.shared.constants import (
    DASHBOARD_DIR,
    ENABLE_DISAGGREGATION,
    EXO_CACHE_HOME,
    EXO_EVENT_LOG_DIR,
    EXO_IMAGE_CACHE_DIR,
    EXO_MAX_CHUNK_SIZE,
    EXO_TRACING_CACHE_DIR,
)
from exo.shared.election import ElectionMessage
from exo.shared.logging import InterceptLogger
from exo.shared.models import model_cards
from exo.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
)
from exo.shared.tracing import TraceEvent, compute_stats, export_trace, load_trace_file
from exo.shared.types.chunks import (
    ErrorChunk,
    ImageChunk,
    InputImageChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.commands import (
    AddCustomModelCard,
    CancelDownload,
    Command,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteDownload,
    DeleteInstance,
    DeleteInstanceLink,
    DownloadCommand,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    SendInputChunk,
    SetInstanceLink,
    SetSharedModelsDirectory,
    SetSharedStorage,
    ShiftInstanceLayers,
    StartDownload,
    TaskCancelled,
    TaskFinished,
    TextGeneration,
)
from exo.shared.types.common import CommandId, Id, NodeId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    InstanceDeleted,
    TracesMerged,
)
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import StageTiming
from exo.shared.types.state import State
from exo.shared.types.storage import SharedStorage
from exo.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from exo.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from exo.shared.types.tasks import ShiftLayers as ShiftLayersTask
from exo.shared.types.tasks import TaskStatus
from exo.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from exo.shared.types.text_generation import (
    Base64ImageHash,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.downloads import DownloadCompleted
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
)
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding
from exo.utils.banner import print_startup_banner
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.disk_event_log import DiskEventLog
from exo.utils.power_sampler import PowerSampler
from exo.utils.state_replica import StateReplica
from exo.utils.task_group import TaskGroup

_API_EVENT_LOG_DIR = EXO_EVENT_LOG_DIR / "api"
ONBOARDING_COMPLETE_FILE = EXO_CACHE_HOME / "onboarding_complete"


class _RebalancePlan(NamedTuple):
    """A rebalance target split plus the migration that would reach it."""

    node_ids: list[NodeId]
    current_layers: dict[NodeId, int]
    node_layers: dict[NodeId, int]
    steps: list[dict[RunnerId, PipelineShardMetadata]]
    stage_timings: Mapping[NodeId, StageTiming]


def _format_to_content_type(image_format: Literal["png", "jpeg", "webp"] | None) -> str:
    return f"image/{image_format or 'png'}"


def _ensure_seed(params: AdvancedImageParams | None) -> AdvancedImageParams:
    """Ensure advanced params has a seed set for distributed consistency."""
    if params is None:
        return AdvancedImageParams(seed=random.randint(0, 2**32 - 1))
    if params.seed is None:
        return params.model_copy(update={"seed": random.randint(0, 2**32 - 1)})
    return params


def _require_disaggregation_enabled() -> None:
    if not ENABLE_DISAGGREGATION:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=(
                "Prefill/decode disaggregation is disabled. "
                "Set ENABLE_DISAGGREGATION=true to enable."
            ),
        )


async def _hugging_face_username(token: str | None) -> str | None:
    """Resolve a token to its Hub username, or None if it is not usable.

    Used both to validate before saving and to show which account is in use.
    Network failures are indistinguishable from bad tokens here, so callers
    treat None as "could not verify" and must not cache it as a hard failure.
    """
    if not token:
        return None
    try:
        async with (
            create_http_session(timeout_profile="short") as session,
            session.get(
                f"{get_hf_endpoint()}/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
            ) as response,
        ):
            if response.status != 200:
                return None
            body = cast(object, await response.json())
            if not isinstance(body, dict):
                return None
            name: object = cast(dict[str, object], body).get("name")
            return name if isinstance(name, str) else None
    except Exception:
        logger.warning("Could not reach Hugging Face to verify token")
        return None


class API:
    def __init__(
        self,
        node_id: NodeId,
        *,
        port: int,
        event_receiver: Receiver[ReplicatedEventDelivery],
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        # This lets us pause the API if an election is running
        election_receiver: Receiver[ElectionMessage],
        state_replica: StateReplica | None = None,
    ) -> None:
        self.state_replica = state_replica
        self._state = State()
        self._event_log = DiskEventLog(_API_EVENT_LOG_DIR)
        self._system_id = SystemId()
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self.event_receiver = event_receiver
        self.election_receiver = election_receiver
        self.node_id: NodeId = node_id
        self.last_completed_election: int = 0
        self.port = port
        self._sent_image_hashes: set[str] = set()

        self.paused: bool = False
        self.paused_ev: anyio.Event = anyio.Event()

        self.app = FastAPI()

        @self.app.middleware("http")
        async def _log_requests(  # pyright: ignore[reportUnusedFunction]
            request: Request,
            call_next: Callable[[Request], Awaitable[StreamingResponse]],
        ) -> Response:
            logger.debug(f"API request: {request.method} {request.url.path}")
            if (
                self.state_replica is not None
                and not self.state_replica.ready
                and request.url.path != "/node_id"
            ):
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Cluster state recovery is in progress"},
                )
            return await call_next(request)

        self._setup_exception_handlers()
        self._setup_cors()
        self._setup_routes()

        self.app.mount(
            "/",
            StaticFiles(
                directory=DASHBOARD_DIR,
                html=True,
            ),
            name="dashboard",
        )

        self._text_generation_queues: dict[
            CommandId,
            Sender[TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk],
        ] = {}
        self._image_generation_queues: dict[
            CommandId, Sender[ImageChunk | ErrorChunk]
        ] = {}
        self._image_store = ImageStore(EXO_IMAGE_CACHE_DIR)
        self._tg: TaskGroup = TaskGroup()

    @property
    def state(self) -> State:
        if self.state_replica is not None:
            return self.state_replica.state
        return self._state

    @state.setter
    def state(self, state: State) -> None:
        if self.state_replica is not None:
            raise RuntimeError("Cannot replace state owned by the shared replica")
        self._state = state

    def reset(
        self,
        result_clock: int,
        event_receiver: Receiver[ReplicatedEventDelivery],
    ) -> None:
        logger.info("Resetting API State")
        self._event_log.close()
        self._event_log = DiskEventLog(_API_EVENT_LOG_DIR)
        if self.state_replica is None:
            self._state = State()
        self._system_id = SystemId()
        self._text_generation_queues = {}
        self._image_generation_queues = {}
        self.unpause(result_clock)
        self.event_receiver.close()
        self.event_receiver = event_receiver
        self._tg.start_soon(self._apply_state)
        self._sent_image_hashes = set()

    def unpause(self, result_clock: int):
        logger.info("Unpausing API")
        self.last_completed_election = result_clock
        self.paused = False
        self.paused_ev.set()
        self.paused_ev = anyio.Event()

    def _setup_exception_handlers(self) -> None:
        self.app.exception_handler(HTTPException)(self.http_exception_handler)

    async def http_exception_handler(
        self, _: Request, exc: HTTPException
    ) -> JSONResponse:
        err = ErrorResponse(
            error=ErrorInfo(
                message=exc.detail,
                type=HTTPStatus(exc.status_code).phrase,
                code=exc.status_code,
            )
        )
        return JSONResponse(err.model_dump(), status_code=exc.status_code)

    def _setup_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        self.app.get("/node_id")(lambda: self.node_id)
        self.app.post("/instance")(self.create_instance)
        self.app.post("/place_instance")(self.place_instance)
        self.app.get("/instance/placement")(self.get_placement)
        self.app.get("/instance/previews")(self.get_placement_previews)
        self.app.get("/instance/await", response_model=None)(self.await_instance)
        self.app.get("/instance/{instance_id}")(self.get_instance)
        self.app.delete("/instance/{instance_id}")(self.delete_instance)
        self.app.post("/instance/{instance_id}/rebalance")(self.rebalance_instance)
        self.app.get("/instance/{instance_id}/rebalance/preview")(
            self.preview_rebalance_instance
        )
        self.app.get("/v1/instance-links")(self.list_instance_links)
        self.app.post("/v1/instance-links")(self.create_instance_link)
        self.app.put("/v1/instance-links/{link_id}")(self.update_instance_link)
        self.app.delete("/v1/instance-links/{link_id}")(self.delete_instance_link)
        self.app.get("/v1/feature-flags")(self.get_feature_flags)
        self.app.get("/models")(self.get_models)
        self.app.get("/v1/models")(self.get_models)
        self.app.post("/models/add")(self.add_custom_model)
        self.app.delete("/models/custom/{model_id:path}")(self.delete_custom_model)
        self.app.get("/models/search")(self.search_models)
        self.app.get("/models/storage")(self.get_models_storage)
        self.app.get("/models/storage/browse")(self.browse_models_storage)
        self.app.put("/models/storage")(self.set_models_storage)
        self.app.put("/models/storage/share")(self.set_models_storage_share)
        self.app.get("/models/storage/network")(self.get_models_storage_network)
        self.app.post("/models/storage/mount")(self.mount_models_storage_share)
        self.app.get("/models/storage/lan")(self.get_models_storage_lan)
        self.app.get("/v1/hf-token")(self.get_hugging_face_token)
        self.app.put("/v1/hf-token")(self.set_hugging_face_token)
        self.app.delete("/v1/hf-token")(self.delete_hugging_face_token)
        self.app.post("/v1/chat/completions", response_model=None)(
            self.chat_completions
        )
        self.app.post("/bench/chat/completions", response_model=None)(
            self.bench_chat_completions
        )
        self.app.post("/v1/images/generations", response_model=None)(
            self.image_generations
        )
        self.app.post("/bench/images/generations")(self.bench_image_generations)
        self.app.post("/v1/images/edits", response_model=None)(self.image_edits)
        self.app.post("/bench/images/edits")(self.bench_image_edits)
        self.app.get("/images")(self.list_images)
        self.app.get("/images/{image_id}")(self.get_image)
        self.app.post("/v1/messages", response_model=None)(self.claude_messages)
        self.app.post("/v1/responses", response_model=None)(self.openai_responses)
        self.app.post("/v1/cancel/{command_id}")(self.cancel_command)

        # Ollama API
        self.app.head("/ollama/")(self.ollama_version)
        self.app.head("/ollama/api/version")(self.ollama_version)
        self.app.post("/ollama/v1/chat/completions", response_model=None)(
            self.chat_completions
        )
        self.app.post("/ollama/api/chat", response_model=None)(self.ollama_chat)
        self.app.post("/ollama/api/api/chat", response_model=None)(self.ollama_chat)
        self.app.post("/ollama/api/v1/chat", response_model=None)(self.ollama_chat)
        self.app.post("/ollama/api/generate", response_model=None)(self.ollama_generate)
        self.app.get("/ollama/api/tags")(self.ollama_tags)
        self.app.get("/ollama/api/api/tags")(self.ollama_tags)
        self.app.get("/ollama/api/v1/tags")(self.ollama_tags)
        self.app.post("/ollama/api/show")(self.ollama_show)
        self.app.get("/ollama/api/ps")(self.ollama_ps)
        self.app.get("/ollama/api/version")(self.ollama_version)

        self.app.get("/state")(self.get_state)
        self.app.get("/state/{path:path}")(self.get_state)
        self.app.get("/events")(self.stream_events)
        self.app.post("/download/start")(self.start_download)
        self.app.delete("/download/{node_id}/{model_id:path}")(self.delete_download)
        self.app.post("/download/cancel")(self.cancel_download)
        self.app.get("/v1/traces")(self.list_traces)
        self.app.post("/v1/traces/delete")(self.delete_traces)
        self.app.get("/v1/traces/{task_id}")(self.get_trace)
        self.app.get("/v1/traces/{task_id}/stats")(self.get_trace_stats)
        self.app.get("/v1/traces/{task_id}/raw")(self.get_trace_raw)
        self.app.get("/onboarding")(self.get_onboarding)
        self.app.post("/onboarding")(self.complete_onboarding)

    def get_state(self, path: str = ""):
        if path == "":
            return self.state
        try:
            x = self.state.model_dump(by_alias=True)
            for attr in path.split("/"):
                if attr != "":
                    if isinstance(x, dict):
                        x = x[attr]  # pyright: ignore[reportUnknownVariableType]
                    elif isinstance(x, list):
                        x = x[int(attr)]  # pyright: ignore[reportUnknownVariableType]
            return cast(Any, x)  # pyright: ignore[reportAny]
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"unable to find path '{path.replace('/', '.')}' in state json",
            ) from e

    async def place_instance(self, payload: PlaceInstanceParams):
        command = PlaceInstance(
            model_card=await ModelCard.load(payload.model_id),
            sharding=payload.sharding,
            instance_meta=payload.instance_meta,
            min_nodes=payload.min_nodes,
            node_layers=payload.node_layers,
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_card=command.model_card,
        )

    async def create_instance(
        self, payload: CreateInstanceParams
    ) -> CreateInstanceResponse:
        instance = payload.instance
        model_card = await ModelCard.load(instance.shard_assignments.model_id)
        required_memory = model_card.storage_size
        available_memory = self._calculate_total_available_memory()

        if required_memory > available_memory:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient memory to create instance. Required: {required_memory.in_gb:.1f}GB, Available: {available_memory.in_gb:.1f}GB",
            )

        command = CreateInstance(
            instance=instance,
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_card=model_card,
        )

    async def get_placement(
        self,
        model_id: ModelId,
        sharding: Sharding = Sharding.Pipeline,
        instance_meta: InstanceMeta = InstanceMeta.MlxRing,
        min_nodes: int = 1,
    ) -> Instance:
        model_card = await ModelCard.load(model_id)

        try:
            placements = get_instance_placements(
                PlaceInstance(
                    model_card=model_card,
                    sharding=sharding,
                    instance_meta=instance_meta,
                    min_nodes=min_nodes,
                ),
                node_memory=node_memory_with_pending_shutdowns(
                    node_memory=self.state.node_memory,
                    tasks=self.state.tasks,
                ),
                node_network=self.state.node_network,
                node_backends=self.state.node_backends,
                topology=self.state.topology,
                current_instances=self.state.instances,
                download_status=self.state.downloads,
                node_rdma_ctl=self.state.node_rdma_ctl,
                node_identities=self.state.node_identities,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        current_ids = set(self.state.instances.keys())
        new_ids = [
            instance_id for instance_id in placements if instance_id not in current_ids
        ]
        if len(new_ids) != 1:
            raise HTTPException(
                status_code=500,
                detail="Expected exactly one new instance from placement",
            )

        return placements[new_ids[0]]

    async def get_placement_previews(
        self,
        model_id: ModelId,
        node_ids: Annotated[list[NodeId] | None, Query()] = None,
    ) -> PlacementPreviewResponse:
        seen: set[tuple[ModelId, Sharding, InstanceMeta, int]] = set()
        previews: list[PlacementPreview] = []
        required_nodes = set(node_ids) if node_ids else None

        if len(list(self.state.topology.list_nodes())) == 0:
            return PlacementPreviewResponse(previews=[])

        try:
            model_card = await ModelCard.load(model_id)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to load model card: {exc}"
            ) from exc
        instance_combinations: list[tuple[Sharding, InstanceMeta, int]] = []
        for sharding in (Sharding.Pipeline, Sharding.Tensor):
            for instance_meta in (InstanceMeta.MlxRing, InstanceMeta.MlxJaccl):
                instance_combinations.extend(
                    [
                        (sharding, instance_meta, i)
                        for i in range(
                            1, len(list(self.state.topology.list_nodes())) + 1
                        )
                    ]
                )
        # TODO: PDD
        # instance_combinations.append((Sharding.PrefillDecodeDisaggregation, InstanceMeta.MlxRing, 1))

        for sharding, instance_meta, min_nodes in instance_combinations:
            try:
                placements = get_instance_placements(
                    PlaceInstance(
                        model_card=model_card,
                        sharding=sharding,
                        instance_meta=instance_meta,
                        min_nodes=min_nodes,
                    ),
                    node_memory=node_memory_with_pending_shutdowns(
                        node_memory=self.state.node_memory,
                        tasks=self.state.tasks,
                    ),
                    node_network=self.state.node_network,
                    node_backends=self.state.node_backends,
                    topology=self.state.topology,
                    current_instances=self.state.instances,
                    required_nodes=required_nodes,
                    download_status=self.state.downloads,
                    node_rdma_ctl=self.state.node_rdma_ctl,
                    node_identities=self.state.node_identities,
                )
            except ValueError as exc:
                if (model_card.model_id, sharding, instance_meta, 0) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=model_card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=None,
                            error=str(exc),
                        )
                    )
                seen.add((model_card.model_id, sharding, instance_meta, 0))
                continue

            current_ids = set(self.state.instances.keys())
            new_instances = [
                instance
                for instance_id, instance in placements.items()
                if instance_id not in current_ids
            ]

            if len(new_instances) != 1:
                if (model_card.model_id, sharding, instance_meta, 0) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=model_card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=None,
                            error="Expected exactly one new instance from placement",
                        )
                    )
                seen.add((model_card.model_id, sharding, instance_meta, 0))
                continue

            instance = new_instances[0]
            shard_assignments = instance.shard_assignments
            placement_node_ids = list(shard_assignments.node_to_runner.keys())

            memory_delta_by_node: dict[str, int] = {}
            if placement_node_ids:
                total_bytes = model_card.storage_size.in_bytes
                per_node = total_bytes // len(placement_node_ids)
                remainder = total_bytes % len(placement_node_ids)
                for index, node_id in enumerate(sorted(placement_node_ids, key=str)):
                    extra = 1 if index < remainder else 0
                    memory_delta_by_node[str(node_id)] = per_node + extra

            if (
                model_card.model_id,
                sharding,
                instance_meta,
                len(placement_node_ids),
            ) not in seen:
                previews.append(
                    PlacementPreview(
                        model_id=model_card.model_id,
                        sharding=sharding,
                        instance_meta=instance_meta,
                        instance=instance,
                        memory_delta_by_node=memory_delta_by_node or None,
                        error=None,
                    )
                )
            seen.add(
                (
                    model_card.model_id,
                    sharding,
                    instance_meta,
                    len(placement_node_ids),
                )
            )

        return PlacementPreviewResponse(previews=previews)

    def get_instance(self, instance_id: InstanceId) -> Instance:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")
        return self.state.instances[instance_id]

    async def await_instance(
        self,
        model_id: ModelId,
        timeout_seconds: float = Query(default=0.0, ge=0.0, le=300.0),
    ) -> StreamingResponse:
        _sleep = 0.1

        async def _stream() -> AsyncGenerator[str, None]:
            deadline = (
                None if timeout_seconds == 0 else anyio.current_time() + timeout_seconds
            )

            while True:
                for instance in self.state.instances.values():
                    if instance.shard_assignments.model_id == model_id:
                        payload = AwaitInstanceReadyMessage(instance=instance)
                        yield f"data: {payload.model_dump_json()}\n\n"
                        return

                if deadline is None:
                    await anyio.sleep(_sleep)
                else:
                    remaining = deadline - anyio.current_time()
                    if remaining <= 0:
                        payload = AwaitInstanceTimeoutMessage(
                            message=f"No instance found for model {model_id}"
                        )
                        yield f"data: {payload.model_dump_json()}\n\n"
                        return

                    await anyio.sleep(min(_sleep, remaining))

        return StreamingResponse(
            with_sse_keepalive(_stream()),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "close",
                "X-Accel-Buffering": "no",
            },
        )

    async def delete_instance(self, instance_id: InstanceId) -> DeleteInstanceResponse:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")

        command = DeleteInstance(
            instance_id=instance_id,
        )
        await self._send(command)
        return DeleteInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            instance_id=instance_id,
        )

    def _plan_rebalance(
        self, instance_id: InstanceId, mode: Literal["speed", "experts"]
    ) -> "_RebalancePlan":
        """Compute the target layer split and the migration steps to reach it.

        Shared by the preview and the live migration so the two can never
        disagree: whatever the preview advertises is exactly what the applier
        would carry out, including the validation that can reject a plan for
        an unsafe intermediate step.
        """
        instance = self.state.instances.get(instance_id)
        if instance is None:
            raise HTTPException(status_code=404, detail="Instance not found")

        shard_assignments = instance.shard_assignments
        node_to_shard = {
            node_id: shard_assignments.runner_to_shard[runner_id]
            for node_id, runner_id in shard_assignments.node_to_runner.items()
        }
        if not all(
            isinstance(shard, PipelineShardMetadata) for shard in node_to_shard.values()
        ):
            raise HTTPException(
                status_code=400,
                detail="Rebalancing is only supported for pipeline-sharded instances",
            )
        if len(node_to_shard) < 2:
            raise HTTPException(
                status_code=400,
                detail="Rebalancing requires an instance spanning at least two nodes",
            )
        missing_memory_nodes = [
            node_id
            for node_id in node_to_shard
            if node_id not in self.state.node_memory
        ]
        if missing_memory_nodes:
            raise HTTPException(
                status_code=400,
                detail=f"No memory report for nodes: {missing_memory_nodes}",
            )

        # Pipeline rank order: the expert-aware split is a contiguous
        # boundary choice, and layers stay attributable to their stages.
        node_ids = sorted(
            node_to_shard.keys(),
            key=lambda node_id: node_to_shard[node_id].device_rank,
        )
        model_card = next(iter(node_to_shard.values())).model_card
        current_layers = {
            node_id: shard.end_layer - shard.start_layer
            for node_id, shard in node_to_shard.items()
        }
        stage_timings = self.state.instance_stage_timings.get(instance_id, {})

        try:
            if mode == "experts":
                node_layers = allocate_layers_by_expert_activity(
                    model_card=model_card,
                    node_ids=node_ids,
                    node_memory=self.state.node_memory,
                    current_layers=current_layers,
                    stage_timings=stage_timings,
                    expert_activity=self.state.instance_expert_activity.get(
                        instance_id, {}
                    ),
                )
            else:
                node_layers = allocate_layers_by_measured_speed(
                    model_card=model_card,
                    node_ids=node_ids,
                    node_memory=self.state.node_memory,
                    current_layers=current_layers,
                    stage_timings=stage_timings,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        planned_steps: list[dict[RunnerId, PipelineShardMetadata]] = []
        if node_layers != current_layers:
            current_pipeline_shards = {
                shard_assignments.node_to_runner[node_id]: shard
                for node_id, shard in node_to_shard.items()
                if isinstance(shard, PipelineShardMetadata)
            }
            target_layer_counts = {
                shard_assignments.node_to_runner[node_id]: layer_count
                for node_id, layer_count in node_layers.items()
            }
            try:
                planned_steps = plan_pipeline_layer_shift_steps(
                    current_pipeline_shards, target_layer_counts
                )
                validate_live_rebalance_steps(
                    model_card=model_card,
                    node_ids=node_ids,
                    node_to_runner=shard_assignments.node_to_runner,
                    node_memory=self.state.node_memory,
                    current_layers=current_layers,
                    steps=planned_steps,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return _RebalancePlan(
            node_ids=node_ids,
            current_layers=current_layers,
            node_layers=node_layers,
            steps=planned_steps,
            stage_timings=stage_timings,
        )

    async def preview_rebalance_instance(
        self, instance_id: InstanceId, mode: Literal["speed", "experts"] = "speed"
    ) -> RebalanceInstanceResponse:
        """Report the split a rebalance would produce, without performing it."""
        return await self.rebalance_instance(instance_id, mode=mode, dry_run=True)

    async def rebalance_instance(
        self,
        instance_id: InstanceId,
        mode: Literal["speed", "experts"] = "speed",
        dry_run: bool = False,
    ) -> RebalanceInstanceResponse:
        """Live-migrate an instance's layers to the measured-speed allocation.

        ``mode="speed"`` splits layers by each stage's measured decode rate;
        ``mode="experts"`` additionally weighs each MoE layer by how much of
        its expert pool recent tokens actually engaged, so layers with
        concentrated routing pack more densely onto a stage.

        Like a disk defragmenter, the rebalance moves one layer at a time
        between adjacent pipeline ranks while the instance keeps serving
        requests; each step pauses generation only for the moment the gaining
        rank loads that layer's weights from local disk.

        With ``dry_run`` the same plan is computed and returned but nothing is
        migrated, so callers can show what a rebalance would actually do
        instead of guessing at it.
        """
        if not dry_run and any(
            isinstance(task, ShiftLayersTask)
            and task.instance_id == instance_id
            and task.task_status
            in {TaskStatus.Pending, TaskStatus.Running, TaskStatus.Complete}
            for task in self.state.tasks.values()
        ):
            raise HTTPException(
                status_code=409,
                detail="Instance already has a layer rebalance in progress",
            )

        plan = self._plan_rebalance(instance_id, mode)

        current_ms = projected_decode_compute_ms(
            node_ids=plan.node_ids,
            layer_counts=plan.current_layers,
            stage_timings=plan.stage_timings,
        )
        projected_ms = projected_decode_compute_ms(
            node_ids=plan.node_ids,
            layer_counts=plan.node_layers,
            stage_timings=plan.stage_timings,
        )
        speedup: float | None = None
        if current_ms is not None and projected_ms is not None and current_ms > 0:
            speedup = (current_ms - projected_ms) / current_ms

        def respond(
            message: str, command_id: CommandId | None
        ) -> RebalanceInstanceResponse:
            return RebalanceInstanceResponse(
                message=message,
                instance_id=instance_id,
                node_layers=plan.node_layers,
                command_id=command_id,
                steps=len(plan.steps),
                dry_run=dry_run,
                current_layers=plan.current_layers,
                current_compute_ms_per_token=current_ms,
                projected_compute_ms_per_token=projected_ms,
                projected_compute_speedup=speedup,
            )

        if plan.node_layers == plan.current_layers:
            return respond(
                "Measured allocation already matches the current layer split.", None
            )
        if dry_run:
            return respond(
                f"Rebalancing would move {len(plan.steps)} layer boundaries.", None
            )

        command = ShiftInstanceLayers(
            instance_id=instance_id,
            node_layers=plan.node_layers,
        )
        await self._send(command)
        return respond("Live layer migration started.", command.command_id)

    async def get_feature_flags(self) -> dict[str, bool]:
        return {"disaggregation": ENABLE_DISAGGREGATION}

    async def list_instance_links(self) -> list[InstanceLink]:
        if not ENABLE_DISAGGREGATION:
            return []
        return list(self.state.instance_links.values())

    async def create_instance_link(
        self, body: InstanceLinkBody
    ) -> InstanceLinkResponse:
        _require_disaggregation_enabled()
        return await self._set_instance_link(InstanceLinkId(), body)

    async def update_instance_link(
        self, link_id: InstanceLinkId, body: InstanceLinkBody
    ) -> InstanceLinkResponse:
        _require_disaggregation_enabled()
        return await self._set_instance_link(link_id, body)

    async def _set_instance_link(
        self, link_id: InstanceLinkId, body: InstanceLinkBody
    ) -> InstanceLinkResponse:
        command = SetInstanceLink(
            link_id=link_id,
            prefill_instances=list(body.prefill_instances),
            decode_instances=list(body.decode_instances),
        )
        await self._send(command)
        return InstanceLinkResponse(
            message="Command received.", command_id=command.command_id
        )

    async def delete_instance_link(
        self, link_id: InstanceLinkId
    ) -> InstanceLinkResponse:
        _require_disaggregation_enabled()
        command = DeleteInstanceLink(link_id=link_id)
        await self._send(command)
        return InstanceLinkResponse(
            message="Command received.", command_id=command.command_id
        )

    async def cancel_command(self, command_id: CommandId) -> CancelCommandResponse:
        """Cancel an active command by closing its stream and notifying workers."""
        sender = self._text_generation_queues.get(
            command_id
        ) or self._image_generation_queues.get(command_id)
        if sender is None:
            raise HTTPException(
                status_code=404,
                detail="Command not found or already completed",
            )

        await self._send(TaskCancelled(cancelled_command_id=command_id))
        sender.close()

        return CancelCommandResponse(
            message="Command cancelled.",
            command_id=command_id,
        )

    async def _token_chunk_stream(
        self, command_id: CommandId
    ) -> AsyncGenerator[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk, None
    ]:
        """Yield chunks for a given command until completion.

        This is the internal low-level stream used by all API adapters.
        """
        try:
            self._text_generation_queues[command_id], recv = channel[
                TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
            ]()

            with recv as token_chunks:
                async for chunk in token_chunks:
                    yield chunk
                    if isinstance(chunk, PrefillProgressChunk):
                        continue
                    if chunk.finish_reason is not None:
                        break

        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._text_generation_queues:
                del self._text_generation_queues[command_id]

    async def _collect_text_generation_with_stats(
        self, command_id: CommandId
    ) -> BenchChatCompletionResponse:
        sampler = PowerSampler(get_node_system=lambda: self.state.node_system)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        model: ModelId | None = None
        finish_reason: FinishReason | None = None

        stats: GenerationStats | None = None

        async with anyio.create_task_group() as tg:
            tg.start_soon(sampler.run)

            async for chunk in self._token_chunk_stream(command_id):
                if isinstance(chunk, PrefillProgressChunk):
                    continue

                sampler.mark_prefill_done()

                if chunk.finish_reason == "error":
                    raise HTTPException(
                        status_code=500,
                        detail=chunk.error_message or "Internal server error",
                    )

                if model is None:
                    model = chunk.model

                if isinstance(chunk, TokenChunk):
                    text_parts.append(chunk.text)

                if isinstance(chunk, ToolCallChunk):
                    tool_calls.extend(
                        ToolCall(
                            id=str(uuid4()),
                            index=i,
                            function=tool,
                        )
                        for i, tool in enumerate(chunk.tool_calls)
                    )

                stats = chunk.stats or stats

                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason

            tg.cancel_scope.cancel()

        combined_text = "".join(text_parts)
        assert model is not None

        return BenchChatCompletionResponse(
            id=command_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=combined_text,
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            generation_stats=stats,
            power_usage=sampler.result(),
        )

    async def _trigger_notify_user_to_download_model(self, model_id: ModelId) -> None:
        logger.warning(
            "TODO: we should send a notification to the user to download the model"
        )

    async def _send_text_generation_with_images(
        self,
        task_params: TextGenerationTaskParams,
        pinned_instance_id: InstanceId | None = None,
    ) -> TextGeneration:
        task_params = task_params.with_card_sampling_defaults()
        images = task_params.images
        if not images:
            command = TextGeneration(
                task_params=task_params, pinned_instance_id=pinned_instance_id
            )
            await self._send(command)
            return command

        hashes = [hashlib.sha256(img.encode("ascii")).hexdigest() for img in images]
        all_hashes = {idx: Base64ImageHash(h) for idx, h in enumerate(hashes)}
        task_params = task_params.model_copy(
            update={"images": [], "image_hashes": all_hashes}
        )
        command = TextGeneration(
            task_params=task_params, pinned_instance_id=pinned_instance_id
        )

        new_images: list[tuple[int, str]] = []
        for idx, (img, h) in enumerate(zip(images, hashes, strict=True)):
            if h not in self._sent_image_hashes:
                self._sent_image_hashes.add(h)
                new_images.append((idx, img))

        if not new_images:
            await self._send(command)
            return command

        all_chunks: list[tuple[int, str]] = []
        for img_idx, img_data in new_images:
            for i in range(0, len(img_data), EXO_MAX_CHUNK_SIZE):
                all_chunks.append((img_idx, img_data[i : i + EXO_MAX_CHUNK_SIZE]))

        for global_idx, (img_idx, chunk_data) in enumerate(all_chunks):
            await self._send(
                SendInputChunk(
                    chunk=InputImageChunk(
                        model=task_params.model,
                        command_id=command.command_id,
                        data=chunk_data,
                        chunk_index=global_idx,
                        total_chunks=len(all_chunks),
                        image_index=img_idx,
                    )
                )
            )

        await self._send(command)
        return command

    async def chat_completions(
        self, payload: ChatCompletionRequest
    ) -> ChatCompletionResponse | StreamingResponse:
        """OpenAI Chat Completions API - adapter."""
        task_params = await chat_request_to_text_generation(payload)
        validated_model, pinned_instance_id = await self._resolve_model_and_instance(
            task_params.model
        )
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(
            task_params, pinned_instance_id
        )

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_chat_stream(
                        command.command_id,
                        self._token_chunk_stream(command.command_id),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_chat_response(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def bench_chat_completions(
        self, payload: BenchChatCompletionRequest
    ) -> BenchChatCompletionResponse | StreamingResponse:
        task_params = await chat_request_to_text_generation(payload)
        validated_model, pinned_instance_id = await self._resolve_model_and_instance(
            task_params.model
        )
        task_params = task_params.model_copy(update={"model": validated_model})

        task_params = task_params.model_copy(
            update={
                "stream": False,
                "bench": True,
                "use_prefix_cache": payload.use_prefix_cache,
            }
        )

        command = await self._send_text_generation_with_images(
            task_params, pinned_instance_id
        )

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_chat_stream(
                        command.command_id,
                        self._token_chunk_stream(command.command_id),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )

        return await self._collect_text_generation_with_stats(command.command_id)

    async def _resolve_model_and_instance(
        self, requested_model: str
    ) -> tuple[ModelId, InstanceId | None]:
        """Resolve a requested model name to a servable model and optional pinned instance.

        Accepts either a plain model id (the master load-balances across all
        instances serving it) or a per-instance alias of the form
        ``<model-id>@<instance-id-prefix>`` which pins the request to that
        exact instance.

        Raises HTTPException (404 unknown, 400 ambiguous prefix); handled by FastAPI.
        """
        model_part, separator, instance_prefix = requested_model.rpartition("@")
        if separator and model_part and instance_prefix:
            model_id = ModelId(model_part)
            matching_instance_ids = [
                instance_id
                for instance_id, instance in self.state.instances.items()
                if instance.shard_assignments.model_id == model_id
                and str(instance_id).startswith(instance_prefix)
            ]
            if not matching_instance_ids:
                raise HTTPException(
                    status_code=404,
                    detail=f"No running instance matches {requested_model}",
                )
            if len(matching_instance_ids) > 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Instance alias {requested_model} is ambiguous",
                )
            return model_id, matching_instance_ids[0]

        model_id = await self._validate_model_has_instance(ModelId(requested_model))
        return model_id, None

    async def _validate_model_has_instance(self, model_id: ModelId) -> ModelId:
        """Validate a model has an active instance.
        If the model isn't even downloaded, triggers notification to user to download model.


        Raises HTTPException 404 if no instance is found for the model.
        """
        if not any(
            instance.shard_assignments.model_id == model_id
            for instance in self.state.instances.values()
        ):
            # Check if model is actually downloaded
            model_is_downloaded = any(
                isinstance(download, DownloadCompleted)
                and download.shard_metadata.model_card.model_id == model_id
                for node_downloads in self.state.downloads.values()
                for download in node_downloads
            )
            if not model_is_downloaded:
                await self._trigger_notify_user_to_download_model(model_id)

            raise HTTPException(
                status_code=404, detail=f"No instance found for model {model_id}"
            )
        return model_id

    def stream_events(self) -> StreamingResponse:
        def _generate_json_array(events: Iterable[Event]) -> Iterable[str]:
            yield "["
            first = True
            for event in events:
                if not first:
                    yield ","
                first = False
                yield event.model_dump_json()
            yield "]"

        return StreamingResponse(
            _generate_json_array(self._event_log.read_all()),
            media_type="application/json",
        )

    async def get_image(self, image_id: str) -> FileResponse:
        stored = self._image_store.get(Id(image_id))
        if stored is None:
            raise HTTPException(status_code=404, detail="Image not found or expired")
        return FileResponse(path=stored.file_path, media_type=stored.content_type)

    async def list_images(self, request: Request) -> ImageListResponse:
        """List all stored images."""
        stored_images = self._image_store.list_images()
        return ImageListResponse(
            data=[
                ImageListItem(
                    image_id=img.image_id,
                    url=self._build_image_url(request, img.image_id),
                    content_type=img.content_type,
                    expires_at=img.expires_at,
                )
                for img in stored_images
            ]
        )

    def _build_image_url(self, request: Request, image_id: Id) -> str:
        host = request.headers.get("host", f"localhost:{self.port}")
        scheme = "https" if request.url.scheme == "https" else "http"
        return f"{scheme}://{host}/v1/images/{image_id}"

    async def image_generations(
        self, request: Request, payload: ImageGenerationTaskParams
    ) -> ImageGenerationResponse | StreamingResponse:
        """Handle image generation requests.

        When stream=True and partial_images > 0, returns a StreamingResponse
        with SSE-formatted events for partial and final images.
        """
        payload = payload.model_copy(
            update={
                "model": await self._validate_model_has_instance(
                    ModelId(payload.model)
                ),
                "advanced_params": _ensure_seed(payload.advanced_params),
            }
        )

        command = ImageGeneration(
            task_params=payload,
        )
        await self._send(command)

        # Check if streaming is requested
        if payload.stream and payload.partial_images and payload.partial_images > 0:
            return StreamingResponse(
                self._generate_image_stream(
                    request=request,
                    command_id=command.command_id,
                    num_images=payload.n or 1,
                    response_format=payload.response_format or "b64_json",
                ),
                media_type="text/event-stream",
            )

        # Non-streaming: collect all image chunks
        return await self._collect_image_generation(
            request=request,
            command_id=command.command_id,
            num_images=payload.n or 1,
            response_format=payload.response_format or "b64_json",
        )

    async def _generate_image_stream(
        self,
        request: Request,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> AsyncGenerator[str, None]:
        """Generate SSE stream of partial and final images."""
        # Track chunks: {(image_index, is_partial): {chunk_index: data}}
        image_chunks: dict[tuple[int, bool], dict[int, str]] = {}
        image_total_chunks: dict[tuple[int, bool], int] = {}
        image_metadata: dict[tuple[int, bool], tuple[int | None, int | None]] = {}
        images_complete = 0

        try:
            self._image_generation_queues[command_id], recv = channel[
                ImageChunk | ErrorChunk
            ]()

            with recv as chunks:
                async for chunk in chunks:
                    if chunk.finish_reason == "error":
                        error_response = ErrorResponse(
                            error=ErrorInfo(
                                message=chunk.error_message or "Internal server error",
                                type="InternalServerError",
                                code=500,
                            )
                        )
                        yield f"data: {error_response.model_dump_json()}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    key = (chunk.image_index, chunk.is_partial)

                    if key not in image_chunks:
                        image_chunks[key] = {}
                        image_total_chunks[key] = chunk.total_chunks
                        image_metadata[key] = (
                            chunk.partial_index,
                            chunk.total_partials,
                        )

                    image_chunks[key][chunk.chunk_index] = chunk.data

                    # Check if this image is complete
                    if len(image_chunks[key]) == image_total_chunks[key]:
                        full_data = "".join(
                            image_chunks[key][i] for i in range(len(image_chunks[key]))
                        )

                        partial_idx, total_partials = image_metadata[key]

                        if chunk.is_partial:
                            # Yield partial image event (always use b64_json for partials)
                            event_data = {
                                "type": "partial",
                                "image_index": chunk.image_index,
                                "partial_index": partial_idx,
                                "total_partials": total_partials,
                                "format": str(chunk.format),
                                "data": {
                                    "b64_json": full_data
                                    if response_format == "b64_json"
                                    else None,
                                },
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"
                        else:
                            # Final image
                            if response_format == "url":
                                image_bytes = base64.b64decode(full_data)
                                content_type = _format_to_content_type(chunk.format)
                                stored = self._image_store.store(
                                    image_bytes, content_type
                                )
                                url = self._build_image_url(request, stored.image_id)
                                event_data = {
                                    "type": "final",
                                    "image_index": chunk.image_index,
                                    "format": str(chunk.format),
                                    "data": {"url": url},
                                }
                            else:
                                event_data = {
                                    "type": "final",
                                    "image_index": chunk.image_index,
                                    "format": str(chunk.format),
                                    "data": {"b64_json": full_data},
                                }
                            yield f"data: {json.dumps(event_data)}\n\n"
                            images_complete += 1

                            if images_complete >= num_images:
                                yield "data: [DONE]\n\n"
                                break

                        # Clean up completed image chunks
                        del image_chunks[key]
                        del image_total_chunks[key]
                        del image_metadata[key]

        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._image_generation_queues:
                del self._image_generation_queues[command_id]

    async def _collect_image_chunks(
        self,
        request: Request | None,
        command_id: CommandId,
        num_images: int,
        response_format: str,
        capture_stats: bool = False,
    ) -> tuple[list[ImageData], ImageGenerationStats | None]:
        """Collect image chunks and optionally capture stats."""
        # Track chunks per image: {image_index: {chunk_index: data}}
        # Only track non-partial (final) images
        image_chunks: dict[int, dict[int, str]] = {}
        image_total_chunks: dict[int, int] = {}
        image_formats: dict[int, Literal["png", "jpeg", "webp"] | None] = {}
        images_complete = 0
        stats: ImageGenerationStats | None = None

        try:
            self._image_generation_queues[command_id], recv = channel[
                ImageChunk | ErrorChunk
            ]()

            while images_complete < num_images:
                with recv as chunks:
                    async for chunk in chunks:
                        if chunk.finish_reason == "error":
                            raise HTTPException(
                                status_code=500,
                                detail=chunk.error_message or "Internal server error",
                            )

                        if chunk.is_partial:
                            continue

                        if chunk.image_index not in image_chunks:
                            image_chunks[chunk.image_index] = {}
                            image_total_chunks[chunk.image_index] = chunk.total_chunks
                            image_formats[chunk.image_index] = chunk.format

                        image_chunks[chunk.image_index][chunk.chunk_index] = chunk.data

                        if capture_stats and chunk.stats is not None:
                            stats = chunk.stats

                        if (
                            len(image_chunks[chunk.image_index])
                            == image_total_chunks[chunk.image_index]
                        ):
                            images_complete += 1

                        if images_complete >= num_images:
                            break

            images: list[ImageData] = []
            for image_idx in range(num_images):
                chunks_dict = image_chunks[image_idx]
                full_data = "".join(chunks_dict[i] for i in range(len(chunks_dict)))
                if response_format == "url" and request is not None:
                    image_bytes = base64.b64decode(full_data)
                    content_type = _format_to_content_type(image_formats.get(image_idx))
                    stored = self._image_store.store(image_bytes, content_type)
                    url = self._build_image_url(request, stored.image_id)
                    images.append(ImageData(b64_json=None, url=url))
                else:
                    images.append(
                        ImageData(
                            b64_json=full_data
                            if response_format == "b64_json"
                            else None,
                            url=None,
                        )
                    )

            return (images, stats if capture_stats else None)
        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._send(TaskFinished(finished_command_id=command_id))
            if command_id in self._image_generation_queues:
                del self._image_generation_queues[command_id]

    async def _collect_image_generation(
        self,
        request: Request,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> ImageGenerationResponse:
        """Collect all image chunks (non-streaming) and return a single response."""
        images, _ = await self._collect_image_chunks(
            request, command_id, num_images, response_format, capture_stats=False
        )
        return ImageGenerationResponse(data=images)

    async def _collect_image_generation_with_stats(
        self,
        request: Request | None,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> BenchImageGenerationResponse:
        sampler = PowerSampler(get_node_system=lambda: self.state.node_system)
        images: list[ImageData] = []
        stats: ImageGenerationStats | None = None
        async with anyio.create_task_group() as tg:
            tg.start_soon(sampler.run)
            images, stats = await self._collect_image_chunks(
                request, command_id, num_images, response_format, capture_stats=True
            )
            tg.cancel_scope.cancel()
        return BenchImageGenerationResponse(
            data=images, generation_stats=stats, power_usage=sampler.result()
        )

    async def bench_image_generations(
        self, request: Request, payload: BenchImageGenerationTaskParams
    ) -> BenchImageGenerationResponse:
        payload = payload.model_copy(
            update={
                "model": await self._validate_model_has_instance(
                    ModelId(payload.model)
                ),
                "stream": False,
                "partial_images": 0,
                "advanced_params": _ensure_seed(payload.advanced_params),
            }
        )

        command = ImageGeneration(
            task_params=payload,
        )
        await self._send(command)

        return await self._collect_image_generation_with_stats(
            request=request,
            command_id=command.command_id,
            num_images=payload.n or 1,
            response_format=payload.response_format or "b64_json",
        )

    async def _send_image_edits_command(
        self,
        image: UploadFile,
        prompt: str,
        model: ModelId,
        n: int,
        size: ImageSize,
        response_format: Literal["url", "b64_json"],
        input_fidelity: Literal["low", "high"],
        stream: bool,
        partial_images: int,
        bench: bool,
        quality: Literal["high", "medium", "low"],
        output_format: Literal["png", "jpeg", "webp"],
        advanced_params: AdvancedImageParams | None,
    ) -> ImageEdits:
        """Prepare and send an image edits command with chunked image upload."""
        validated_model = await self._validate_model_has_instance(model)
        advanced_params = _ensure_seed(advanced_params)

        image_content = await image.read()
        image_data = base64.b64encode(image_content).decode("utf-8")

        image_strength = 0.7 if input_fidelity == "high" else 0.3

        data_chunks = [
            image_data[i : i + EXO_MAX_CHUNK_SIZE]
            for i in range(0, len(image_data), EXO_MAX_CHUNK_SIZE)
        ]
        total_chunks = len(data_chunks)

        command = ImageEdits(
            task_params=ImageEditsTaskParams(
                image_data="",
                total_input_chunks=total_chunks,
                prompt=prompt,
                model=validated_model,
                n=n,
                size=size,
                response_format=response_format,
                image_strength=image_strength,
                stream=stream,
                partial_images=partial_images,
                bench=bench,
                quality=quality,
                output_format=output_format,
                advanced_params=advanced_params,
            ),
        )

        logger.info(
            f"Sending input image: {len(image_data)} bytes in {total_chunks} chunks"
        )
        for chunk_index, chunk_data in enumerate(data_chunks):
            await self._send(
                SendInputChunk(
                    chunk=InputImageChunk(
                        model=validated_model,
                        command_id=command.command_id,
                        data=chunk_data,
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                    )
                )
            )

        await self._send(command)
        return command

    async def image_edits(
        self,
        request: Request,
        image: UploadFile = File(...),  # noqa: B008
        prompt: str = Form(...),
        model: str = Form(...),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: Literal["url", "b64_json"] = Form("b64_json"),
        input_fidelity: Literal["low", "high"] = Form("low"),
        stream: str = Form("false"),
        partial_images: str = Form("0"),
        quality: Literal["high", "medium", "low"] = Form("medium"),
        output_format: Literal["png", "jpeg", "webp"] = Form("png"),
        advanced_params: str | None = Form(None),
    ) -> ImageGenerationResponse | StreamingResponse:
        """Handle image editing requests (img2img)."""
        # Parse string form values to proper types
        stream_bool = stream.lower() in ("true", "1", "yes")
        partial_images_int = int(partial_images) if partial_images.isdigit() else 0

        parsed_advanced_params: AdvancedImageParams | None = None
        if advanced_params:
            with contextlib.suppress(Exception):
                parsed_advanced_params = AdvancedImageParams.model_validate_json(
                    advanced_params
                )

        command = await self._send_image_edits_command(
            image=image,
            prompt=prompt,
            model=ModelId(model),
            n=n,
            size=normalize_image_size(size),
            response_format=response_format,
            input_fidelity=input_fidelity,
            stream=stream_bool,
            partial_images=partial_images_int,
            bench=False,
            quality=quality,
            output_format=output_format,
            advanced_params=parsed_advanced_params,
        )

        if stream_bool and partial_images_int > 0:
            return StreamingResponse(
                self._generate_image_stream(
                    request=request,
                    command_id=command.command_id,
                    num_images=n,
                    response_format=response_format,
                ),
                media_type="text/event-stream",
            )

        return await self._collect_image_generation(
            request=request,
            command_id=command.command_id,
            num_images=n,
            response_format=response_format,
        )

    async def bench_image_edits(
        self,
        request: Request,
        image: UploadFile = File(...),  # noqa: B008
        prompt: str = Form(...),
        model: str = Form(...),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: Literal["url", "b64_json"] = Form("b64_json"),
        input_fidelity: Literal["low", "high"] = Form("low"),
        quality: Literal["high", "medium", "low"] = Form("medium"),
        output_format: Literal["png", "jpeg", "webp"] = Form("png"),
        advanced_params: str | None = Form(None),
    ) -> BenchImageGenerationResponse:
        """Handle benchmark image editing requests with generation stats."""
        parsed_advanced_params: AdvancedImageParams | None = None
        if advanced_params:
            with contextlib.suppress(Exception):
                parsed_advanced_params = AdvancedImageParams.model_validate_json(
                    advanced_params
                )

        command = await self._send_image_edits_command(
            image=image,
            prompt=prompt,
            model=ModelId(model),
            n=n,
            size=normalize_image_size(size),
            response_format=response_format,
            input_fidelity=input_fidelity,
            stream=False,
            partial_images=0,
            bench=True,
            quality=quality,
            output_format=output_format,
            advanced_params=parsed_advanced_params,
        )

        return await self._collect_image_generation_with_stats(
            request=request,
            command_id=command.command_id,
            num_images=n,
            response_format=response_format,
        )

    async def claude_messages(
        self, payload: ClaudeMessagesRequest
    ) -> ClaudeMessagesResponse | StreamingResponse:
        """Claude Messages API - adapter."""
        task_params = await claude_request_to_text_generation(payload)
        validated_model, pinned_instance_id = await self._resolve_model_and_instance(
            task_params.model
        )
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(
            task_params, pinned_instance_id
        )

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_claude_stream(
                        command.command_id,
                        payload.model,
                        self._token_chunk_stream(command.command_id),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_claude_response(
                    command.command_id,
                    payload.model,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def openai_responses(
        self, payload: ResponsesRequest
    ) -> ResponsesResponse | StreamingResponse:
        """OpenAI Responses API."""
        task_params = await responses_request_to_text_generation(payload)
        validated_model, pinned_instance_id = await self._resolve_model_and_instance(
            task_params.model
        )
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(
            task_params, pinned_instance_id
        )

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_responses_stream(
                        command.command_id,
                        payload.model,
                        self._token_chunk_stream(command.command_id),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            return StreamingResponse(
                collect_responses_response(
                    command.command_id,
                    payload.model,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def _ollama_root(self) -> JSONResponse:
        """Respond to HEAD / from Ollama CLI connectivity checks."""
        return JSONResponse(content="Ollama is running")

    async def ollama_chat(
        self, request: Request
    ) -> OllamaChatResponse | StreamingResponse:
        """Ollama Chat API — accepts JSON regardless of Content-Type."""
        body = await request.body()
        payload = OllamaChatRequest.model_validate_json(body)
        task_params = ollama_request_to_text_generation(payload)
        validated_model, pinned_instance_id = await self._resolve_model_and_instance(
            task_params.model
        )
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(
            task_params, pinned_instance_id
        )

        if payload.stream:
            return StreamingResponse(
                generate_ollama_chat_stream(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_ollama_chat_response(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def ollama_generate(
        self, request: Request
    ) -> OllamaGenerateResponse | StreamingResponse:
        """Ollama Generate API — accepts JSON regardless of Content-Type."""
        body = await request.body()
        payload = OllamaGenerateRequest.model_validate_json(body)
        task_params = ollama_generate_request_to_text_generation(payload)
        validated_model, pinned_instance_id = await self._resolve_model_and_instance(
            task_params.model
        )
        task_params = task_params.model_copy(update={"model": validated_model})

        command = await self._send_text_generation_with_images(
            task_params, pinned_instance_id
        )

        if payload.stream:
            return StreamingResponse(
                generate_ollama_generate_stream(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_ollama_generate_response(
                    command.command_id,
                    self._token_chunk_stream(command.command_id),
                ),
                media_type="application/json",
            )

    async def ollama_tags(self) -> OllamaTagsResponse:
        """Returns list of models in Ollama tags format. We return the downloaded ones only."""

        downloaded_model_ids: set[ModelId] = set()
        for node_downloads in self.state.downloads.values():
            for dl in node_downloads:
                if isinstance(dl, DownloadCompleted):
                    downloaded_model_ids.add(dl.shard_metadata.model_card.model_id)

        cards = [
            c
            for c in await model_cards.card_cache.list_all()
            if c.model_id in downloaded_model_ids
        ]

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return OllamaTagsResponse(
            models=[
                OllamaModelTag(
                    name=str(card.model_id),
                    model=str(card.model_id),
                    modified_at=now,
                    size=card.storage_size.in_bytes,
                    digest="sha256:000000000000",
                    details=OllamaModelDetails(
                        family=card.family or None,
                        quantization_level=card.quantization or None,
                    ),
                )
                for card in cards
            ]
        )

    async def ollama_show(self, request: Request) -> OllamaShowResponse:
        """Returns model information in Ollama show format."""
        body = await request.body()
        payload = OllamaShowRequest.model_validate_json(body)
        model_name = payload.name or payload.model
        if not model_name:
            raise HTTPException(status_code=400, detail="name or model is required")
        try:
            card = await ModelCard.load(ModelId(model_name))
        except Exception as exc:
            raise HTTPException(
                status_code=404, detail=f"Model not found: {model_name}"
            ) from exc

        capabilities: list[OllamaCapability] = []
        if ModelTask.TextGeneration in card.tasks:
            capabilities.extend(("completion", "tools"))
        if card.vision is not None:
            capabilities.append("vision")

        architecture = card.family or "unknown"
        model_info: dict[str, Any] = {
            "general.architecture": architecture,
            "general.basename": card.base_model or str(card.model_id),
        }
        if card.context_length > 0:
            model_info[f"{architecture}.context_length"] = card.context_length

        return OllamaShowResponse(
            modelfile=f"FROM {card.model_id}",
            template="{{ .Prompt }}",
            details=OllamaModelDetails(
                family=card.family or None,
                quantization_level=card.quantization or None,
            ),
            model_info=model_info,
            capabilities=capabilities,
        )

    async def ollama_ps(self) -> OllamaPsResponse:
        """Returns list of running models (active instances)."""
        models: list[OllamaPsModel] = []
        seen: set[str] = set()
        for instance in self.state.instances.values():
            model_id = str(instance.shard_assignments.model_id)
            if model_id in seen:
                continue
            seen.add(model_id)
            models.append(
                OllamaPsModel(
                    name=model_id,
                    model=model_id,
                    size=0,
                )
            )
        return OllamaPsResponse(models=models)

    async def ollama_version(self) -> dict[str, str]:
        """Returns version information for Ollama API compatibility."""
        return {"version": "1.0.0"}

    def _calculate_total_available_memory(self) -> Memory:
        """Calculate total available memory across all nodes in bytes."""
        total_available = Memory()

        for memory in self.state.node_memory.values():
            total_available += memory.ram_available

        return total_available

    async def get_models(self, status: str | None = Query(default=None)) -> ModelList:
        """Returns list of available models, optionally filtered by being downloaded."""
        cards = await model_cards.card_cache.list_all()

        if status == "downloaded":
            downloaded_model_ids: set[str] = set()
            for node_downloads in self.state.downloads.values():
                for dl in node_downloads:
                    if isinstance(dl, DownloadCompleted):
                        downloaded_model_ids.add(dl.shard_metadata.model_card.model_id)
            cards = [c for c in cards if c.model_id in downloaded_model_ids]

        data = [
            ModelListModel(
                id=card.model_id,
                hugging_face_id=card.model_id,
                name=card.model_id.short(),
                description="",
                tags=[],
                storage_size_megabytes=card.storage_size.in_mb,
                supports_tensor=card.supports_tensor,
                tasks=[task.value for task in card.tasks],
                is_custom=card.is_custom,
                family=card.family,
                quantization=card.quantization,
                base_model=card.base_model,
                capabilities=card.capabilities,
                reasoning_dialect=card.reasoning_dialect,
                context_length=card.context_length,
            )
            for card in cards
        ]
        data.extend(self._running_instance_alias_models())
        return ModelList(data=data)

    def _running_instance_alias_models(self) -> list[ModelListModel]:
        """One model entry per running instance, addressable via its alias.

        The alias id (``<model-id>@<instance-id-prefix>``) can be used as the
        ``model`` field on any text endpoint to pin requests to that instance.
        """
        alias_models: list[ModelListModel] = []
        for instance_id, instance in self.state.instances.items():
            model_id = instance.shard_assignments.model_id
            card = model_cards.card_cache.get(model_id)
            alias_models.append(
                ModelListModel(
                    id=f"{model_id}@{str(instance_id)[:8]}",
                    hugging_face_id=model_id,
                    name=f"{model_id.short()} @{str(instance_id)[:8]}",
                    description=f"Running instance of {model_id}",
                    instance_id=str(instance_id),
                    tasks=[task.value for task in card.tasks] if card else [],
                    capabilities=card.capabilities if card else [],
                    reasoning_dialect=card.reasoning_dialect if card else "none",
                    context_length=card.context_length if card else 0,
                )
            )
        return alias_models

    async def add_custom_model(self, payload: AddCustomModelParams) -> ModelListModel:
        """Fetch a model from HuggingFace and save as a custom model card, then sync across the cluster."""
        try:
            card = await ModelCard.fetch_from_hf(payload.model_id)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to fetch model: {exc}"
            ) from exc

        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=AddCustomModelCard(model_card=card),
            )
        )

        # Immediately update the local cache so the subsequent GET /models
        # returns the new model without waiting for the event round-trip.
        model_cards.card_cache.cc[card.model_id] = card

        return ModelListModel(
            id=card.model_id,
            hugging_face_id=card.model_id,
            name=card.model_id.short(),
            description="",
            tags=[],
            storage_size_megabytes=int(card.storage_size.in_mb),
            supports_tensor=card.supports_tensor,
            tasks=[task.value for task in card.tasks],
            is_custom=True,
        )

    async def delete_custom_model(self, model_id: ModelId) -> JSONResponse:
        """Delete a user-added custom model card and sync deletion across the cluster."""
        card = model_cards.card_cache.get(model_id)
        if card is None or not card.is_custom:
            raise HTTPException(status_code=404, detail="Custom model card not found")

        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=DeleteCustomModelCard(model_id=model_id),
            )
        )

        return JSONResponse(
            {"message": "Model card deleted", "model_id": str(model_id)}
        )

    async def search_models(
        self, query: str = "", limit: int = 20
    ) -> list[HuggingFaceSearchResult]:
        """Search HuggingFace Hub — tries mlx-community first, falls back to all of HuggingFace."""
        from huggingface_hub import ModelInfo, list_models

        def _to_results(models: Iterable[ModelInfo]) -> list[HuggingFaceSearchResult]:
            return [
                HuggingFaceSearchResult(
                    id=m.id,
                    author=m.author or "",
                    downloads=m.downloads or 0,
                    likes=m.likes or 0,
                    last_modified=str(m.last_modified or ""),
                    tags=list(m.tags or []),
                )
                for m in models
            ]

        # Search mlx-community first
        mlx_results = _to_results(
            list_models(
                search=query or None,
                author="mlx-community",
                sort="downloads",
                limit=limit,
            )
        )
        if mlx_results:
            return mlx_results

        # Fall back to searching all of HuggingFace
        return _to_results(
            list_models(
                search=query or None,
                sort="downloads",
                limit=limit,
            )
        )

    async def run(self):
        shutdown_ev = anyio.Event()

        try:
            async with self._tg as tg:
                logger.info("Starting API")
                tg.start_soon(self._apply_state)
                tg.start_soon(self._pause_on_new_election)
                tg.start_soon(self._cleanup_expired_images)
                print_startup_banner(self.port)
                tg.start_soon(self.run_api, shutdown_ev)
                try:
                    await anyio.sleep_forever()
                finally:
                    with anyio.CancelScope(shield=True):
                        # IMPORTANT: when new queues are added, update this (for proper shutdown semantics)
                        self._shutdown_queues(self._text_generation_queues)
                        self._shutdown_queues(self._image_generation_queues)

                        shutdown_ev.set()
        finally:
            self._event_log.close()
            self.command_sender.close()
            self.event_receiver.close()

    @staticmethod
    def _shutdown_queues[K, V](queues: dict[K, Sender[V]]):
        for v in queues.values():
            v.close()

    async def run_api(self, ev: anyio.Event):
        cfg = Config()
        cfg.bind = [f"0.0.0.0:{self.port}"]
        # nb: shared.logging needs updating if any of this changes
        cfg.accesslog = None
        cfg.errorlog = "-"
        cfg.logger_class = InterceptLogger

        # prevents hangs when mid-request and connection refuses to close
        cfg.graceful_timeout = 2  # seconds
        cfg.shutdown_timeout = 3  # seconds

        with anyio.CancelScope(shield=True):
            try:
                await serve(
                    cast(ASGIFramework, self.app),
                    cfg,
                    shutdown_trigger=ev.wait,
                )
                if not ev.is_set():
                    raise ShutdownError(
                        "Server exited without shutdown trigger - exiting abnormally"
                    )
            except LifespanTimeoutError as e:
                logger.warning(
                    "Graceful server shutdown timed out, some connections forcebly closed"
                )
                logger.opt(exception=e).debug("")

    async def _apply_state(self):
        with self.event_receiver as events:
            async for delivery in events:
                i_event = delivery.indexed_event
                self._event_log.append(i_event.event)
                if self.state_replica is None:
                    self._state = apply(self._state, i_event)
                event = i_event.event

                if isinstance(event, ChunkGenerated):
                    if queue := self._image_generation_queues.get(
                        event.command_id, None
                    ):
                        assert isinstance(event.chunk, ImageChunk)
                        try:
                            await queue.send(event.chunk)
                        except (BrokenResourceError, ClosedResourceError):
                            self._image_generation_queues.pop(event.command_id, None)
                    if queue := self._text_generation_queues.get(
                        event.command_id, None
                    ):
                        assert not isinstance(event.chunk, ImageChunk)
                        try:
                            await queue.send(event.chunk)
                        except (BrokenResourceError, ClosedResourceError):
                            self._text_generation_queues.pop(event.command_id, None)
                if isinstance(event, InstanceDeleted):
                    self._close_streams_for_instance(
                        event.instance_id,
                        delivery.state_after_event,
                    )
                if isinstance(event, TracesMerged):
                    self._save_merged_trace(event)

    def _close_streams_for_instance(
        self,
        instance_id: InstanceId,
        event_state: State | None = None,
    ) -> None:
        """Close any active generation streams for commands running on the given instance."""
        state = self.state if event_state is None else event_state
        for task in state.tasks.values():
            if task.instance_id != instance_id:
                continue
            if not isinstance(
                task, (TextGenerationTask, ImageGenerationTask, ImageEditsTask)
            ):
                continue
            if sender := self._text_generation_queues.pop(task.command_id, None):
                sender.close()
            if sender := self._image_generation_queues.pop(task.command_id, None):
                sender.close()

    def _save_merged_trace(self, event: TracesMerged) -> None:
        traces = [
            TraceEvent(
                name=t.name,
                start_us=t.start_us,
                duration_us=t.duration_us,
                rank=t.rank,
                category=t.category,
            )
            for t in event.traces
        ]
        output_path = EXO_TRACING_CACHE_DIR / f"trace_{event.task_id}.json"
        export_trace(traces, output_path)
        logger.debug(f"Saved merged trace to {output_path}")

    async def _pause_on_new_election(self):
        with self.election_receiver as ems:
            async for message in ems:
                if message.clock > self.last_completed_election:
                    self.paused = True

    async def _cleanup_expired_images(self):
        """Periodically clean up expired images from the store."""
        cleanup_interval_seconds = 300  # 5 minutes
        while True:
            await anyio.sleep(cleanup_interval_seconds)
            removed = self._image_store.cleanup_expired()
            if removed > 0:
                logger.debug(f"Cleaned up {removed} expired images")

    async def _send(self, command: Command):
        while self.paused:
            await self.paused_ev.wait()
        await self.command_sender.send(
            ForwarderCommand(origin=self._system_id, command=command)
        )

    async def _send_download(self, command: DownloadCommand):
        await self.download_command_sender.send(
            ForwarderDownloadCommand(origin=self._system_id, command=command)
        )

    async def get_models_storage(self) -> ModelsStorageResponse:
        """Report shared storage: the share (or legacy path) and per-node health.

        Every node in the cluster is listed, not just the ones that answered, so
        a node missing an entry in the share is visible as such rather than
        silently absent.
        """
        share = self.state.shared_storage
        statuses = self.state.shared_models_dir_statuses
        node_ids = sorted({*self.state.topology.list_nodes(), *statuses})
        per_node: list[ModelsStorageNodeStatus] = []
        for node_id in node_ids:
            status = statuses.get(node_id)
            mount_path = share.path_for(node_id) if share is not None else None
            if status is None and mount_path is None and share is not None:
                per_node.append(
                    ModelsStorageNodeStatus(
                        node_id=node_id,
                        valid=False,
                        error="No path configured for this node",
                    )
                )
                continue
            if status is None:
                continue
            per_node.append(
                ModelsStorageNodeStatus(
                    node_id=node_id,
                    valid=status.valid,
                    error=status.error,
                    free_bytes=status.free_bytes,
                    path=status.path,
                    mount_path=mount_path,
                    writable=status.writable,
                )
            )
        return ModelsStorageResponse(
            path=self.state.shared_models_dir,
            per_node=per_node,
            share=(
                None
                if share is None
                else ModelsStorageShare(
                    share_id=share.share_id,
                    mounts=dict(share.mounts),
                    source=share.source,
                    label=share.label,
                )
            ),
            mode=(
                "share"
                if share is not None
                else ("path" if self.state.shared_models_dir else "none")
            ),
        )

    async def get_models_storage_network(
        self,
        server: Annotated[str | None, Query()] = None,
    ) -> ModelsStorageNetworkResponse:
        """Discover LAN file servers, or list one server's guest shares.

        Runs on the node serving the dashboard. Discovery is best-effort;
        a host that does not announce itself can still be queried directly.
        """
        if server is not None:
            try:
                shares = await to_thread.run_sync(list_lan_shares, server)
            except ShareMountError as list_error:
                return ModelsStorageNetworkResponse(
                    server=server, error=str(list_error)
                )
            listed: list[ModelsStorageNetworkShare] = []
            for share in shares:
                mounted_at = (
                    await to_thread.run_sync(existing_nfs_mount_point, share.uri)
                    if share.protocol == "nfs"
                    else None
                )
                listed.append(
                    ModelsStorageNetworkShare(
                        name=share.name,
                        uri=share.uri,
                        protocol=share.protocol,
                        host=server,
                        mounted_at=str(mounted_at) if mounted_at else None,
                    )
                )
            return ModelsStorageNetworkResponse(server=server, shares=listed)
        servers = await to_thread.run_sync(discover_smb_servers)
        return ModelsStorageNetworkResponse(
            servers=[
                ModelsStorageNetworkServer(host=found.host, name=found.name)
                for found in servers
            ]
        )

    async def set_models_storage_share(
        self, payload: SetModelsStorageShareParams
    ) -> SetModelsStorageShareResponse:
        """Define the named share, or clear it by sending no ``share_id``.

        Node paths are stored exactly as given: they are deliberately not
        compared with each other, since the whole point of the share is that a
        Linux mount and a macOS mount of the same export have different paths.
        """
        share_id = payload.share_id.strip() if payload.share_id else None
        if not share_id:
            command = SetSharedStorage(storage=None)
            await self._send(command)
            return SetModelsStorageShareResponse(
                command_id=command.command_id, share=None
            )

        mounts = {
            node_id: path.strip()
            for node_id, path in payload.mounts.items()
            if path.strip()
        }
        current = self.state.shared_storage
        storage = SharedStorage(
            share_id=share_id,
            mounts=mounts,
            source=payload.source.strip() if payload.source else None,
            label=payload.label.strip() if payload.label else None,
            # Outrank whatever any node still has on disk, or a stale copy
            # resurfacing after an election would overwrite this edit.
            revision=(current.revision + 1) if current is not None else 1,
        )
        command = SetSharedStorage(storage=storage)
        await self._send(command)
        return SetModelsStorageShareResponse(
            command_id=command.command_id,
            share=ModelsStorageShare(
                share_id=storage.share_id,
                mounts=dict(storage.mounts),
                source=storage.source,
                label=storage.label,
            ),
        )

    async def set_models_storage(
        self, payload: SetModelsStorageParams
    ) -> SetModelsStorageResponse:
        path_text = payload.path.strip() if payload.path is not None else None
        if path_text == "":
            path_text = None
        command = SetSharedModelsDirectory(path=path_text)
        await self._send(command)
        return SetModelsStorageResponse(command_id=command.command_id, path=path_text)

    async def browse_models_storage(
        self,
        path: Annotated[str | None, Query()] = None,
        include_hidden: Annotated[bool, Query()] = False,
    ) -> ModelsStorageBrowseResponse:
        """List directories for the Shared Model Storage folder picker.

        Browses the API node's whole local filesystem, mounted network volumes
        included — an empty path returns the filesystem root plus shortcuts, and
        every directory is reachable from there. The chosen path must exist at
        the same location on every node.
        """
        # With shared storage configured, an empty path means "the share",
        # not this node's filesystem root: the browser must show what the
        # cluster will actually read, not whichever disk happens to back the
        # dashboard. Falls back to the local roots when this node cannot
        # resolve the share, so the picker still works during setup.
        target = path
        if target is None or target.strip() == "":
            # The holder knows the path this node actually validated —
            # including one the worker auto-mounted, which the share's mount
            # map never contains.
            installed = get_shared_models_dir()
            resolved = (
                str(installed)
                if installed is not None
                else resolve_shared_models_path(
                    self.state.shared_storage,
                    self.state.shared_models_dir,
                    self.node_id,
                )
            )
            if resolved:
                target = resolved
        result = browse_shared_models_directories(target, include_hidden=include_hidden)
        return ModelsStorageBrowseResponse(
            path=result.path,
            parent_path=result.parent_path,
            entries=[
                ModelsStorageBrowseEntry(
                    name=entry.name, path=entry.path, hidden=entry.hidden
                )
                for entry in result.entries
            ],
            error=result.error,
            truncated=result.truncated,
            network_volumes=[
                ModelsStorageNetworkVolume(
                    path=volume.path,
                    source=volume.source,
                    filesystem=volume.filesystem,
                    reachable=volume.reachable,
                    label=volume.label,
                )
                for volume in result.network_volumes
            ],
        )

    async def get_models_storage_lan(self) -> ModelsStorageBrowseResponse:
        """What LAN servers offer that this node has not mounted.

        Slow by nature — discovery plus per-server listings — so it is its own
        endpoint the picker loads alongside the instant folder listing, with
        every server probed concurrently and a hard ceiling on the whole scan.
        """
        volumes = await to_thread.run_sync(list_network_volumes)
        mounted = " ".join(volume.source.lower() for volume in volumes)
        available: list[ModelsStorageNetworkShare] = []
        try:
            servers = await to_thread.run_sync(
                discover_smb_servers, abandon_on_cancel=True
            )
        except ShareMountError:
            servers = ()

        async def probe(host: str) -> None:
            try:
                offered = await to_thread.run_sync(
                    list_lan_shares, host, abandon_on_cancel=True
                )
            except ShareMountError:
                return
            for share in offered:
                tail = share.uri.split("://", 1)[-1].lower()
                export = "/" + tail.split("/", 1)[-1]
                if host in mounted and (
                    export in mounted or share.name.lower() in mounted
                ):
                    continue
                available.append(
                    ModelsStorageNetworkShare(
                        name=share.name,
                        uri=share.uri,
                        protocol=share.protocol,
                        host=host,
                    )
                )

        with anyio.move_on_after(15):
            async with anyio.create_task_group() as scan_group:
                for found in servers[:6]:
                    scan_group.start_soon(probe, found.host)

        return ModelsStorageBrowseResponse(
            path="",
            parent_path=None,
            entries=[],
            available_shares=sorted(
                available, key=lambda share: (share.host, share.name.lower())
            ),
        )

    async def mount_models_storage_share(
        self, payload: MountShareParams
    ) -> MountShareResponse:
        """Attach a guest SMB share to THIS node and return where it landed.

        Turns "available on the LAN" into "browsable here" in one click. NFS
        is refused with the reason: mounting it needs root, which exo lacks.
        """
        try:
            mounted = await to_thread.run_sync(ensure_share_mounted, payload.uri)
        except ShareMountError as mount_error:
            return MountShareResponse(path=None, error=str(mount_error))
        return MountShareResponse(path=str(mounted))

    async def get_hugging_face_token(self) -> HuggingFaceTokenResponse:
        """Report token status for THIS node only.

        The token is deliberately kept out of cluster state: /state is
        unauthenticated, and events are gossiped to peers and persisted to the
        event log, so a token there would leak three ways. Each node therefore
        holds its own, and the response never contains the token itself.
        """
        source = await get_hf_token_source()
        token = await get_hf_token()
        return HuggingFaceTokenResponse(
            configured=token is not None,
            source=source,
            hint=mask_hf_token(token) if token else None,
            username=await _hugging_face_username(token) if token else None,
        )

    async def set_hugging_face_token(
        self, payload: SetHuggingFaceTokenParams
    ) -> SetHuggingFaceTokenResponse:
        token = payload.token.strip()
        if not token:
            raise HTTPException(status_code=400, detail="Token must not be empty")

        username = await _hugging_face_username(token)
        if username is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Hugging Face rejected this token. Check it was copied in full "
                    "from https://huggingface.co/settings/tokens and has read access."
                ),
            )

        await set_hf_token(token)
        # An existing HF_TOKEN env var shadows the file we just wrote, so the
        # save silently would not take effect. Say so instead.
        env_shadowed = os.environ.get("HF_TOKEN") is not None
        return SetHuggingFaceTokenResponse(
            configured=True,
            source="env" if env_shadowed else "file",
            hint=mask_hf_token(token),
            username=username,
            warning=(
                "Saved, but the HF_TOKEN environment variable is set on this node "
                "and takes precedence. Unset it for the saved token to be used."
                if env_shadowed
                else None
            ),
        )

    async def delete_hugging_face_token(self) -> HuggingFaceTokenResponse:
        _ = await delete_hf_token()
        source = await get_hf_token_source()
        token = await get_hf_token()
        return HuggingFaceTokenResponse(
            configured=token is not None,
            source=source,
            hint=mask_hf_token(token) if token else None,
            username=None,
        )

    async def start_download(
        self, payload: StartDownloadParams
    ) -> StartDownloadResponse:
        command = StartDownload(
            target_node_id=payload.target_node_id,
            shard_metadata=payload.shard_metadata,
        )
        await self._send_download(command)
        return StartDownloadResponse(command_id=command.command_id)

    async def delete_download(
        self, node_id: NodeId, model_id: ModelId
    ) -> DeleteDownloadResponse:
        command = DeleteDownload(
            target_node_id=node_id,
            model_id=ModelId(model_id),
        )
        await self._send_download(command)
        return DeleteDownloadResponse(command_id=command.command_id)

    async def cancel_download(
        self,
        payload: CancelDownloadParams,
    ) -> CancelDownloadResponse:
        command = CancelDownload(
            target_node_id=payload.target_node_id,
            model_id=payload.model_id,
        )
        await self._send_download(command)
        return CancelDownloadResponse(command_id=command.command_id)

    @staticmethod
    def _get_trace_path(task_id: str) -> Path:
        trace_path = EXO_TRACING_CACHE_DIR / f"trace_{task_id}.json"
        if not trace_path.resolve().is_relative_to(EXO_TRACING_CACHE_DIR.resolve()):
            raise HTTPException(status_code=400, detail=f"Invalid task ID: {task_id}")
        return trace_path

    async def list_traces(self) -> TraceListResponse:
        traces: list[TraceListItem] = []

        for trace_file in sorted(
            EXO_TRACING_CACHE_DIR.glob("trace_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            # Extract task_id from filename (trace_{task_id}.json)
            task_id = trace_file.stem.removeprefix("trace_")
            stat = trace_file.stat()
            created_at = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()
            traces.append(
                TraceListItem(
                    task_id=task_id,
                    created_at=created_at,
                    file_size=stat.st_size,
                )
            )

        return TraceListResponse(traces=traces)

    async def get_trace(self, task_id: str) -> TraceResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        trace_events = load_trace_file(trace_path)

        return TraceResponse(
            task_id=task_id,
            traces=[
                TraceEventResponse(
                    name=event.name,
                    start_us=event.start_us,
                    duration_us=event.duration_us,
                    rank=event.rank,
                    category=event.category,
                )
                for event in trace_events
            ],
        )

    async def get_trace_stats(self, task_id: str) -> TraceStatsResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        trace_events = load_trace_file(trace_path)
        stats = compute_stats(trace_events)

        return TraceStatsResponse(
            task_id=task_id,
            total_wall_time_us=stats.total_wall_time_us,
            by_category={
                category: TraceCategoryStats(
                    total_us=cat_stats.total_us,
                    count=cat_stats.count,
                    min_us=cat_stats.min_us,
                    max_us=cat_stats.max_us,
                    avg_us=cat_stats.avg_us,
                )
                for category, cat_stats in stats.by_category.items()
            },
            by_rank={
                rank: TraceRankStats(
                    by_category={
                        category: TraceCategoryStats(
                            total_us=cat_stats.total_us,
                            count=cat_stats.count,
                            min_us=cat_stats.min_us,
                            max_us=cat_stats.max_us,
                            avg_us=cat_stats.avg_us,
                        )
                        for category, cat_stats in rank_stats.items()
                    }
                )
                for rank, rank_stats in stats.by_rank.items()
            },
        )

    async def get_trace_raw(self, task_id: str) -> FileResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        return FileResponse(
            path=trace_path,
            media_type="application/json",
            filename=f"trace_{task_id}.json",
        )

    async def delete_traces(self, request: DeleteTracesRequest) -> DeleteTracesResponse:
        deleted: list[str] = []
        not_found: list[str] = []
        for task_id in request.task_ids:
            trace_path = self._get_trace_path(task_id)
            if trace_path.exists():
                trace_path.unlink()
                deleted.append(task_id)
            else:
                not_found.append(task_id)
        return DeleteTracesResponse(deleted=deleted, not_found=not_found)

    async def get_onboarding(self) -> JSONResponse:
        return JSONResponse({"completed": ONBOARDING_COMPLETE_FILE.exists()})

    async def complete_onboarding(self) -> JSONResponse:
        ONBOARDING_COMPLETE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ONBOARDING_COMPLETE_FILE.write_text("true")
        return JSONResponse({"completed": True})
