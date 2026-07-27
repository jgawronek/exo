"""Per-layer MoE expert activation recording.

Mixture-of-experts models route every token through a small subset of each
MoE layer's experts. Which experts fire is the only per-layer activity signal
that varies during decode (dense layers are hit uniformly), so recording it
enables expert-aware layer rebalancing: a layer whose recent tokens
concentrated on few experts touches a smaller weight working set than one
whose routing spread widely.

The recording hook wraps ``SwitchGLU``/``SwitchMLP`` — the single dispatch
point every mlx_lm MoE architecture funnels its selected expert indices
through — and accumulates a lazily-evaluated histogram per decoder layer.
The accumulation adds only graph nodes to the step being built; the histogram
is materialised when the runner drains it on its stage-timing cadence, never
inside the decode hot path.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast, final

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU, SwitchMLP

from exo.shared.types.profiling import LayerExpertActivity
from exo.worker.runner.bootstrap import logger

# Expert dispatches whose flattened index count exceeds this are prefill
# sweeps (many tokens at once); recording them would swamp the decode signal
# the rebalance cares about, so they are skipped. Decode dispatches are
# (concurrent generations x top_k experts) indices, comfortably below this.
_MAX_DECODE_DISPATCH_INDICES = 256

# Most-selected experts kept per layer for display purposes.
_TOP_ACTIVATIONS_KEPT = 8

_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@final
@dataclass
class ExpertActivityRecorder:
    """Accumulates per-layer expert selection histograms between drains."""

    _layer_index_by_module: dict[int, int] = field(default_factory=dict)
    _num_experts_by_module: dict[int, int] = field(default_factory=dict)
    _counts_by_layer: dict[int, mx.array] = field(default_factory=dict)
    _tokens_by_layer: dict[int, int] = field(default_factory=dict)

    def register_model(self, model: object, start_layer: int) -> int:
        """Map the model's MoE dispatch modules to absolute layer indices.

        The local (possibly pipeline-sliced) decoder layer list is re-indexed
        from zero, so ``start_layer`` restores the absolute layer number.
        Returns the number of MoE layers found; zero for dense models.
        """
        self._layer_index_by_module.clear()
        self._num_experts_by_module.clear()
        self._counts_by_layer.clear()
        self._tokens_by_layer.clear()
        if not isinstance(model, nn.Module):
            # Tests drive the builder with stand-in models; there is nothing
            # to record on them.
            return 0
        registered_layers: set[int] = set()
        named_modules = cast(
            list[tuple[str, nn.Module]],
            model.named_modules(),  # pyright: ignore[reportUnknownMemberType]
        )
        for name, module in named_modules:
            if not isinstance(module, (SwitchGLU, SwitchMLP)):
                continue
            match = _LAYER_INDEX_PATTERN.search(name)
            if match is None:
                continue
            absolute_layer = start_layer + int(match.group(1))
            up_projection = cast(object, getattr(module, "up_proj", None))
            num_experts = getattr(up_projection, "num_experts", None)
            if not isinstance(num_experts, int) or num_experts < 2:
                continue
            self._layer_index_by_module[id(module)] = absolute_layer
            self._num_experts_by_module[id(module)] = num_experts
            registered_layers.add(absolute_layer)
        if registered_layers:
            logger.info(
                f"expert activity recording enabled for {len(registered_layers)} "
                f"MoE layers (absolute layers "
                f"{min(registered_layers)}-{max(registered_layers)})"
            )
        return len(registered_layers)

    def record(self, module: nn.Module, indices: mx.array) -> None:
        layer_index = self._layer_index_by_module.get(id(module))
        if layer_index is None:
            return
        if indices.size == 0 or indices.size > _MAX_DECODE_DISPATCH_INDICES:
            return
        num_experts = self._num_experts_by_module[id(module)]
        histogram = (
            mx.zeros(num_experts, dtype=mx.uint32).at[indices.reshape(-1)].add(1)
        )
        previous = self._counts_by_layer.get(layer_index)
        self._counts_by_layer[layer_index] = (
            histogram if previous is None else previous + histogram
        )
        top_k = indices.shape[-1] if indices.ndim > 0 else 1
        self._tokens_by_layer[layer_index] = self._tokens_by_layer.get(
            layer_index, 0
        ) + max(1, indices.size // max(1, top_k))

    def drain(self) -> dict[int, LayerExpertActivity]:
        """Materialise and reset the accumulated histograms.

        Called by the runner between decode steps, so forcing evaluation here
        never interrupts a step's lazy graph.
        """
        if not self._counts_by_layer:
            return {}
        mx.eval(*self._counts_by_layer.values())
        drained: dict[int, LayerExpertActivity] = {}
        for layer_index, counts_array in self._counts_by_layer.items():
            counts = cast(list[int], counts_array.tolist())
            total = sum(counts)
            if total <= 0:
                continue
            entropy = 0.0
            unique = 0
            for count in counts:
                if count <= 0:
                    continue
                unique += 1
                probability = count / total
                entropy -= probability * math.log(probability)
            top = sorted(
                ((index, count) for index, count in enumerate(counts) if count > 0),
                key=lambda pair: -pair[1],
            )[:_TOP_ACTIVATIONS_KEPT]
            drained[layer_index] = LayerExpertActivity(
                num_experts=counts_array.size,
                tokens_measured=self._tokens_by_layer.get(layer_index, 0),
                unique_experts_activated=unique,
                effective_experts=math.exp(entropy),
                top_activations={str(index): count for index, count in top},
            )
        self._counts_by_layer.clear()
        self._tokens_by_layer.clear()
        return drained


expert_activity = ExpertActivityRecorder()


_DispatchCall = Callable[[nn.Module, mx.array, mx.array], mx.array]

_dispatch_recording_patched = False


def patch_switch_dispatch_recording() -> None:
    """Wrap the MoE dispatch modules to record selected expert indices."""
    global _dispatch_recording_patched
    if _dispatch_recording_patched:
        return
    _dispatch_recording_patched = True
    original_glu_call = cast(_DispatchCall, SwitchGLU.__call__)
    original_mlp_call = cast(_DispatchCall, SwitchMLP.__call__)

    def recording_glu_call(self: nn.Module, x: mx.array, indices: mx.array) -> mx.array:
        expert_activity.record(self, indices)
        return original_glu_call(self, x, indices)

    def recording_mlp_call(self: nn.Module, x: mx.array, indices: mx.array) -> mx.array:
        expert_activity.record(self, indices)
        return original_mlp_call(self, x, indices)

    SwitchGLU.__call__ = recording_glu_call
    SwitchMLP.__call__ = recording_mlp_call
