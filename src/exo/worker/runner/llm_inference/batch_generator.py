import itertools
import time
from collections import deque
from collections.abc import Generator, Iterator, Mapping
from dataclasses import dataclass, field
from typing import BinaryIO

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.download.download_utils import build_model_path
from exo.shared.constants import EXO_MAX_CONCURRENT_REQUESTS
from exo.shared.types.chunks import ErrorChunk, GenerationChunk, PrefillProgressChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.profiling import DecodeTimingSample, LayerExpertActivity
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    GenerationTask,
    ShiftLayers,
    TaskId,
    TextGeneration,
)
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    GenerationResponse,
)
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import MpReceiver, MpSender
from exo.utils.step_profiler import StepProfiler
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Engine
from exo.worker.engines.mlx.auto_parallel import decode_timings, shift_pipeline_layers
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.disaggregated.adapter import write_cache_to_wire
from exo.worker.engines.mlx.disaggregated.serve import run_prefill_for_request
from exo.worker.engines.mlx.expert_activity import expert_activity
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.generator.generate import (
    PrefillCancelled,
    mlx_generate,
    warmup_inference,
)
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    mx_all_gather_tasks,
    mx_any,
)
from exo.worker.engines.mlx.vision import VisionProcessor
from exo.worker.runner.bootstrap import logger

from .model_output_parsers import apply_all_parsers, map_responses_to_chunks
from .tool_parsers import ToolParser


class GeneratorQueue[T]:
    def __init__(self):
        self._q = deque[T]()

    def push(self, t: T):
        self._q.append(t)

    def gen(self) -> Generator[T | None]:
        while True:
            if len(self._q) == 0:
                yield None
            else:
                yield self._q.popleft()


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"

# During steady-state decode every engine step used to issue a task-agreement
# collective even when no rank had pending work; over a TCP ring that costs
# 1-3ms out of a 40-100ms token budget. Steps are rank-lockstep while the
# engine has work, so agreeing on a fixed step cadence keeps collective counts
# matched across ranks while amortizing the cost. New requests arriving
# mid-decode wait at most this many tokens before joining the batch.
TASK_AGREEMENT_INTERVAL_STEPS = 8

# Same idea during prefill: the per-chunk progress callback used to run the
# cancel + task agreement collectives on every chunk (3 collectives per
# chunk). Smaller wavefront chunks made that relatively expensive.
PREFILL_AGREEMENT_INTERVAL_CHUNKS = 4


def _check_for_debug_prompts(task_params: TextGenerationTaskParams) -> None:
    """Check for debug prompt triggers in the input."""
    from exo.worker.engines.mlx.utils_mlx import mlx_force_oom

    if len(task_params.input) == 0:
        return
    prompt = task_params.input[0].content
    if not prompt:
        return
    if EXO_RUNNER_MUST_FAIL in prompt:
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        mlx_force_oom()
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)


@dataclass(eq=False)
class SequentialGenerator(Engine):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    vision_processor: VisionProcessor | None = None
    check_for_cancel_every: int = 50

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration] = field(default_factory=list, init=False)
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _active: (
        tuple[
            TextGeneration,
            # mlx generator that does work
            Generator[GenerationResponse],
            # queue that the 1st generator should push to and 3rd generator should pull from
            GeneratorQueue[GenerationResponse],
            # generator to get parsed outputs
            Iterator[GenerationChunk | None],
        ]
        | None
    ) = field(default=None, init=False)

    def warmup(self):
        self.check_for_cancel_every = warmup_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
        )

    def submit(
        self,
        task: GenerationTask,
    ) -> None:
        assert isinstance(task, TextGeneration)
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all)."""
        agreed, different = mx_all_gather_tasks(self._maybe_queue, self.group)
        # Extend from `agreed` (sorted by task_id on all ranks) to guarantee every
        # rank enqueues tasks in the same order, preventing TP collective deadlocks.
        self._queue.extend(agreed)
        self._maybe_queue = list(different)

    def agree_on_cancellations(self) -> None:
        """Agree between all ranks about which tasks to cancel."""
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])

        if mx_any(has_cancel_all, self.group):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, self.group)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)

    def step(
        self,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | FinishedResponse | CancelledResponse]
    ]:
        if self._active is None:
            self.agree_on_tasks()

            if self._queue:
                self._start_next()
            else:
                return map(
                    lambda task: (task, CancelledResponse()), self._cancelled_tasks
                )

        assert self._active is not None

        task, gen, queue, output_generator = self._active
        output: list[
            tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
        ] = []
        try:
            response = next(gen)
            queue.push(response)
            # drain potentially many responses every time
            while (parsed := next(output_generator, None)) is not None:
                output.append((task.task_id, parsed))

        except (StopIteration, PrefillCancelled):
            output.append((task.task_id, FinishedResponse()))
            self._active = None
            if self._queue:
                self._start_next()

        except Exception as e:
            self._send_error(task, e)
            self._active = None
            raise

        return filter(
            lambda chunk: (
                not isinstance(chunk[1], GenerationChunk) or self.device_rank == 0
            ),
            itertools.chain(
                output,
                map(lambda task: (task, CancelledResponse()), self._cancelled_tasks),
            ),
        )

    def _start_next(self) -> None:
        task = self._queue.popleft()
        try:
            gen = self._build_generator(task)
        except Exception as e:
            self._send_error(task, e)
            raise
        queue = GeneratorQueue[GenerationResponse]()

        if task.task_params.bench:
            output_generator: Iterator[GenerationChunk | None] = map(
                lambda r: map_responses_to_chunks(r, self.model_id), queue.gen()
            )
        else:
            output_generator = apply_all_parsers(
                queue.gen(),
                apply_chat_template(self.tokenizer, task.task_params),
                self.tool_parser,
                self.tokenizer,
                type(self.model),
                self.model_id,
                task.task_params.tools,
            )
        self._active = (task, gen, queue, output_generator)

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        if self.device_rank == 0:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            )

    def _build_generator(self, task: TextGeneration) -> Generator[GenerationResponse]:
        _check_for_debug_prompts(task.task_params)
        prompt = apply_chat_template(self.tokenizer, task.task_params)

        def on_prefill_progress(processed: int, total: int) -> None:
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=PrefillProgressChunk(
                            model=self.model_id,
                            processed_tokens=processed,
                            total_tokens=total,
                        ),
                    )
                )

        def distributed_prompt_progress_callback() -> None:
            self.agree_on_cancellations()
            if self.should_cancel(task.task_id):
                raise PrefillCancelled()

            self.agree_on_tasks()

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                self.agree_on_cancellations()
                if self.should_cancel(task.task_id):
                    raise PrefillCancelled()

                self.agree_on_tasks()

        return mlx_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            task=task.task_params,
            prompt=prompt,
            kv_prefix_cache=self.kv_prefix_cache,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
            group=self.group,
            vision_processor=self.vision_processor,
        )

    def close(self) -> None:
        del self.model, self.tokenizer, self.group

    def poll_decode_timing(self) -> DecodeTimingSample | None:
        return decode_timings.drain_sample()

    def poll_expert_activity(self) -> Mapping[int, LayerExpertActivity]:
        return expert_activity.drain()

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        cache = run_prefill_for_request(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            request=request,
        )
        write_cache_to_wire(
            wfile,
            cache,
            request_id=request.request_id,
            model_id=request.model_id,
            start_pos=request.start_pos,
        )


@dataclass(eq=False)
class BatchGenerator(Engine):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    tool_parser: ToolParser | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    check_for_cancel_every: int = 50
    vision_processor: VisionProcessor | None = None
    draft_model: Model | None = None
    pipeline_shard: PipelineShardMetadata | None = None

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[TextGeneration | ShiftLayers] = field(
        default_factory=list, init=False
    )
    _maybe_cancel: list[TextGeneration] = field(default_factory=list, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _shift_queue: deque[ShiftLayers] = field(default_factory=deque, init=False)
    _pending_shift: ShiftLayers | None = field(default=None, init=False)
    _gen: ExoBatchGenerator = field(init=False)
    _profiler: StepProfiler = field(init=False)
    _steps_until_task_agreement: int = field(default=0, init=False)
    _active_tasks: dict[
        int,
        tuple[
            TextGeneration,
            GeneratorQueue[GenerationResponse],
            Iterator[GenerationChunk | None],
        ],
    ] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._profiler = StepProfiler(f"engine-rank{self.device_rank}")
        self._gen = ExoBatchGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            vision_processor=self.vision_processor,
            draft_model=self.draft_model,
        )

    def warmup(self):
        self.check_for_cancel_every = warmup_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            model_id=self.model_id,
        )

    def submit(
        self,
        task: GenerationTask,
    ) -> None:
        assert isinstance(task, TextGeneration)
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._maybe_queue.append(task)

    def submit_shard_update(self, task: ShiftLayers) -> bool:
        if self.group is None or self.pipeline_shard is None:
            return False
        self._maybe_queue.append(task)
        return True

    def _apply_shard_update(self, task: ShiftLayers) -> None:
        """Apply an agreed boundary shift now that the batch is idle.

        Runs on every rank at the same logical step (the shift came out of
        the rank-agreed queue and batch drain is lockstep across ranks), so
        no rank can be mid-forward while layers move.
        """
        assert self.group is not None and self.pipeline_shard is not None
        new_shard = next(
            shard
            for shard in task.new_shards.values()
            if shard.device_rank == self.device_rank
        )
        boundaries_changed = (new_shard.start_layer, new_shard.end_layer) != (
            self.pipeline_shard.start_layer,
            self.pipeline_shard.end_layer,
        )
        if boundaries_changed:
            shift_pipeline_layers(
                self.model,
                self.group,
                old_shard=self.pipeline_shard,
                new_shard=new_shard,
                model_path=build_model_path(self.model_id),
            )
        self.pipeline_shard = new_shard
        # Stored prefixes hold KV for the previous layer range; clear on every
        # rank (even no-op ones) so restore-length agreement stays trivial.
        if self.kv_prefix_cache is not None:
            self.kv_prefix_cache.clear()
        # Discard timing accumulated under the old layer layout.
        decode_timings.drain_sample()

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all)."""
        agreed, different = mx_all_gather_tasks(self._maybe_queue, self.group)
        # Extend from `agreed` (sorted by task_id on all ranks) to guarantee every
        # rank enqueues tasks in the same order, preventing TP collective deadlocks.
        for agreed_task in agreed:
            if isinstance(agreed_task, ShiftLayers):
                self._shift_queue.append(agreed_task)
            else:
                self._queue.append(agreed_task)
        self._maybe_queue = list(different)

    def agree_on_cancellations(self) -> None:
        """Agree between all ranks about which tasks to cancel."""
        has_cancel_all = False
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                has_cancel_all = True
                continue
            if task_id in self._all_tasks:
                self._maybe_cancel.append(self._all_tasks[task_id])

        if mx_any(has_cancel_all, self.group):
            self._cancelled_tasks.add(CANCEL_ALL_TASKS)

        agreed, different = mx_all_gather_tasks(self._maybe_cancel, self.group)
        self._cancelled_tasks.update(task.task_id for task in agreed)
        self._maybe_cancel = list(different)

    def step(
        self,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
    ]:
        if not self._queue:
            # Rank-consistent cadence: the countdown only advances on steps
            # where the queue is empty, and steps are lockstep across ranks
            # while the engine has work, so every rank agrees on the same
            # steps. An idle engine agrees immediately so freshly submitted
            # tasks start without waiting out the countdown.
            self._steps_until_task_agreement -= 1
            if self._steps_until_task_agreement <= 0 or not self._gen.has_work:
                with self._profiler.bucket("agree_on_tasks"):
                    self.agree_on_tasks()
                self._steps_until_task_agreement = TASK_AGREEMENT_INTERVAL_STEPS

        if self._pending_shift is None and self._shift_queue:
            self._pending_shift = self._shift_queue.popleft()

        if (
            self._pending_shift is not None
            and not self._active_tasks
            and not self._gen.has_work
        ):
            shift = self._pending_shift
            self._pending_shift = None
            self._apply_shard_update(shift)
            return iter([(shift.task_id, FinishedResponse())])

        # Submit any queued tasks to the engine. Submission pauses while a
        # shard update waits for the batch to drain, so the update cannot be
        # starved under sustained load; paused tasks resume right after it.
        while (
            self._queue
            and self._pending_shift is None
            and len(self._active_tasks) < EXO_MAX_CONCURRENT_REQUESTS
        ):
            task = self._queue.popleft()
            try:
                uid = self._start_task(task)
            except PrefillCancelled:
                continue
            except Exception as e:
                self._send_error(task, e)
                raise

            queue = GeneratorQueue[GenerationResponse]()
            if self.device_rank != 0:
                # Non-rank-0 chunks are discarded below, so parsing and
                # detokenizing them is wasted per-token CPU work.
                output_generator: Iterator[GenerationChunk | None] = iter(())
            elif task.task_params.bench:
                output_generator = map(
                    lambda r: map_responses_to_chunks(r, self.model_id), queue.gen()
                )
            else:
                output_generator = apply_all_parsers(
                    queue.gen(),
                    apply_chat_template(self.tokenizer, task.task_params),
                    self.tool_parser,
                    self.tokenizer,
                    type(self.model),
                    self.model_id,
                    task.task_params.tools,
                )
            self._active_tasks[uid] = (task, queue, output_generator)

        if not self._gen.has_work:
            return self._apply_cancellations()

        with self._profiler.bucket("mlx_step"):
            results = self._gen.step()

        output: list[
            tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
        ] = []
        with self._profiler.bucket("parse_detokenize"):
            for uid, response in results:
                if uid not in self._active_tasks:
                    # should we error here?
                    logger.warning(f"{uid=} not found in active tasks")
                    continue

                task, queue, output_generator = self._active_tasks[uid]
                if self.device_rank == 0:
                    queue.push(response)
                    # If a generator fails to parse for some reason and returns
                    # early, we should not crash
                    while (parsed := next(output_generator, None)) is not None:
                        output.append((task.task_id, parsed))

                # check if original response was terminal and append a Finished()
                if response.finish_reason is not None:
                    output.append((task.task_id, FinishedResponse()))
                    del self._active_tasks[uid]

        with self._profiler.bucket("apply_cancellations"):
            cancellations = list(self._apply_cancellations())

        self._profiler.end_step(len(output))

        return filter(
            lambda chunk: (
                not isinstance(chunk[1], GenerationChunk) or self.device_rank == 0
            ),
            itertools.chain(output, cancellations),
        )

    def _apply_cancellations(
        self,
    ) -> Iterator[tuple[TaskId, CancelledResponse]]:
        if not self._cancelled_tasks:
            return iter([])

        cancel_all = CANCEL_ALL_TASKS in self._cancelled_tasks

        uids_to_cancel: list[int] = []
        results: list[tuple[TaskId, CancelledResponse]] = []

        for uid, (task, _, _) in list(self._active_tasks.items()):
            if task.task_id in self._cancelled_tasks or cancel_all:
                uids_to_cancel.append(uid)
                results.append((task.task_id, CancelledResponse()))
                del self._active_tasks[uid]

        if uids_to_cancel:
            self._gen.cancel(uids_to_cancel)

        already_cancelled = {tid for tid, _ in results}
        for tid in self._cancelled_tasks:
            if tid != CANCEL_ALL_TASKS and tid not in already_cancelled:
                results.append((tid, CancelledResponse()))

        self._cancelled_tasks.clear()
        return iter(results)

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        if self.device_rank == 0:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=ErrorChunk(
                        model=self.model_id,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            )

    def _start_task(self, task: TextGeneration) -> int:
        _check_for_debug_prompts(task.task_params)
        prompt = apply_chat_template(self.tokenizer, task.task_params)

        def on_prefill_progress(processed: int, total: int) -> None:
            if self.device_rank == 0:
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=task.command_id,
                        chunk=PrefillProgressChunk(
                            model=self.model_id,
                            processed_tokens=processed,
                            total_tokens=total,
                        ),
                    )
                )

        prefill_chunks_since_agreement = 0

        def distributed_prompt_progress_callback() -> None:
            # Every rank invokes this callback the same number of times per
            # prefill (real + wavefront filler iterations), so gating on the
            # invocation count keeps collective counts matched while paying
            # the cancel/task agreement cost once per few chunks instead of
            # per chunk.
            nonlocal prefill_chunks_since_agreement
            prefill_chunks_since_agreement += 1
            if prefill_chunks_since_agreement < PREFILL_AGREEMENT_INTERVAL_CHUNKS:
                return
            prefill_chunks_since_agreement = 0
            self.agree_on_cancellations()
            if self.should_cancel(task.task_id):
                raise PrefillCancelled()

            self.agree_on_tasks()

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                self.agree_on_cancellations()
                if self.should_cancel(task.task_id):
                    self._cancelled_tasks.add(task.task_id)
                # Task agreement is handled on a step cadence in step();
                # repeating it here would just add collectives per token.

        return self._gen.submit(
            task_params=task.task_params,
            prompt=prompt,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
        )

    def close(self) -> None:
        self._gen.close()
        del self.model, self.tokenizer, self.group

    def poll_decode_timing(self) -> DecodeTimingSample | None:
        return decode_timings.drain_sample()

    def poll_expert_activity(self) -> Mapping[int, LayerExpertActivity]:
        return expert_activity.drain()

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        cache = run_prefill_for_request(
            model=self.model,
            tokenizer=self.tokenizer,
            group=self.group,
            kv_prefix_cache=self.kv_prefix_cache,
            request=request,
        )
        write_cache_to_wire(
            wfile,
            cache,
            request_id=request.request_id,
            model_id=request.model_id,
            start_pos=request.start_pos,
        )
