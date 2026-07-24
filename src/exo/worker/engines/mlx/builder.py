import contextlib
import os
from collections.abc import Generator
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_model

from exo.download.download_utils import build_model_path
from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    SequentialGenerator,
)
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser

from .cache import KVPrefixCache
from .types import Model
from .utils_mlx import (
    initialize_mlx,
    load_mlx_items,
)
from .vision import VisionProcessor


def load_speculative_draft_model(group: mx.distributed.Group | None) -> Model | None:
    """Load the EXO_DRAFT_MODEL weights for speculative decoding, if configured.

    The draft model must already be downloaded locally and must share the
    target model's tokenizer. Any failure is handled here by disabling
    speculative decoding rather than failing the runner, since the target
    model alone is sufficient to serve requests.
    """
    draft_model_id = os.environ.get("EXO_DRAFT_MODEL")
    if not draft_model_id:
        return None
    if group is not None:
        logger.warning(
            "EXO_DRAFT_MODEL is set but speculative decoding only supports "
            "single-node instances; ignoring the draft model"
        )
        return None
    try:
        model_path = build_model_path(ModelId(draft_model_id))
        draft_model, _ = load_model(model_path, lazy=False, strict=False)
    except Exception:
        logger.opt(exception=True).warning(
            f"Failed to load draft model {draft_model_id}; "
            "speculative decoding disabled"
        )
        return None
    logger.info(f"Loaded draft model {draft_model_id} for speculative decoding")
    return cast(Model, draft_model)


@dataclass
class MlxBuilder(Builder):
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    inference_model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    group: mx.distributed.Group | None = None
    vision_processor: VisionProcessor | None = None
    draft_model: Model | None = None

    def connect(self, bound_instance: BoundInstance) -> None:
        self.group = initialize_mlx(bound_instance)

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        (
            self.inference_model,
            self.tokenizer,
            self.vision_processor,
        ) = yield from load_mlx_items(bound_instance, self.group)
        self.draft_model = load_speculative_draft_model(self.group)

    def close(self) -> None:
        with contextlib.suppress(NameError, AttributeError):
            del self.inference_model
        with contextlib.suppress(NameError, AttributeError):
            del self.tokenizer
        with contextlib.suppress(NameError, AttributeError):
            del self.group
        with contextlib.suppress(NameError, AttributeError):
            del self.draft_model

    def build(
        self,
    ) -> Engine:
        assert self.inference_model
        assert self.tokenizer

        vision_processor = self.vision_processor

        tool_parser = None
        logger.info(
            f"model has_tool_calling={self.tokenizer.has_tool_calling} using tokens {self.tokenizer.tool_call_start}, {self.tokenizer.tool_call_end}"
        )
        if (
            self.tokenizer.tool_call_start
            and self.tokenizer.tool_call_end
            and self.tokenizer.tool_parser  # type: ignore
        ):
            tool_parser = make_mlx_parser(
                self.tokenizer.tool_call_start,
                self.tokenizer.tool_call_end,
                self.tokenizer.tool_parser,  # type: ignore
            )

        kv_prefix_cache = KVPrefixCache(self.group)

        device_rank = 0 if self.group is None else self.group.rank()
        if os.environ.get("EXO_NO_BATCH"):
            logger.info("using SequentialGenerator (batching disabled)")
            return SequentialGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
            )
        else:
            logger.info("using BatchGenerator")
            return BatchGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
                draft_model=self.draft_model,
            )
