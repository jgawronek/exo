import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU

from exo.worker.engines.mlx.expert_activity import (
    ExpertActivityRecorder,
    patch_switch_dispatch_recording,
)


class _MoeLayer(nn.Module):
    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.switch_mlp = SwitchGLU(
            input_dims=8, hidden_dims=16, num_experts=num_experts
        )


class _TinyMoeModel(nn.Module):
    def __init__(self, num_layers: int, num_experts: int) -> None:
        super().__init__()
        self.layers = [_MoeLayer(num_experts) for _ in range(num_layers)]


def test_register_model_maps_moe_layers_to_absolute_indices() -> None:
    recorder = ExpertActivityRecorder()
    model = _TinyMoeModel(num_layers=3, num_experts=4)

    registered = recorder.register_model(model, start_layer=10)

    assert registered == 3
    recorder.record(model.layers[1].switch_mlp, mx.array([[0, 3]]))
    drained = recorder.drain()
    assert set(drained) == {11}
    assert drained[11].activations == [1, 0, 0, 1]
    assert drained[11].num_experts == 4
    assert drained[11].tokens_measured == 1


def test_record_accumulates_across_steps_and_resets_on_drain() -> None:
    recorder = ExpertActivityRecorder()
    model = _TinyMoeModel(num_layers=1, num_experts=4)
    recorder.register_model(model, start_layer=0)
    dispatch = model.layers[0].switch_mlp

    recorder.record(dispatch, mx.array([[1, 2]]))
    recorder.record(dispatch, mx.array([[1, 3]]))

    drained = recorder.drain()
    assert drained[0].activations == [0, 2, 1, 1]
    assert drained[0].tokens_measured == 2
    assert recorder.drain() == {}


def test_record_skips_prefill_sized_dispatches() -> None:
    recorder = ExpertActivityRecorder()
    model = _TinyMoeModel(num_layers=1, num_experts=4)
    recorder.register_model(model, start_layer=0)

    prefill_indices = mx.zeros((512, 2), dtype=mx.uint32)
    recorder.record(model.layers[0].switch_mlp, prefill_indices)

    assert recorder.drain() == {}


def test_unregistered_module_is_ignored() -> None:
    recorder = ExpertActivityRecorder()
    model = _TinyMoeModel(num_layers=1, num_experts=4)

    recorder.record(model.layers[0].switch_mlp, mx.array([[0, 1]]))

    assert recorder.drain() == {}


def test_patched_dispatch_records_and_preserves_output() -> None:
    patch_switch_dispatch_recording()
    from exo.worker.engines.mlx.expert_activity import expert_activity

    model = _TinyMoeModel(num_layers=1, num_experts=4)
    expert_activity.register_model(model, start_layer=5)

    x = mx.random.normal((1, 1, 8))
    indices = mx.array([[[0, 2]]])
    output = model.layers[0].switch_mlp(x, indices)
    mx.eval(output)

    # SwitchGLU returns one output per selected expert; the caller applies
    # the routing weights.
    assert output.shape == (1, 1, 2, 8)
    drained = expert_activity.drain()
    assert drained[5].activations == [1, 0, 1, 0]
