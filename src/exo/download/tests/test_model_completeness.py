"""Completeness + stale-share detection for local model directories.

Covers two robustness fixes: a partial copy that has weight files but not the
small config the loader reads first must not read as complete (it would crash
the runner at load), and a share-backed model that vanishes must be detectable
as stale so its cached completion can be re-fetched instead of crash-looping.
"""

import json
from pathlib import Path

from exo.download.download_utils import (
    is_model_directory_complete,
    share_completion_is_stale,
)

_INDEX = json.dumps(
    {
        "metadata": {"total_size": 200},
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        },
    }
)


def _write_weights(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.safetensors.index.json").write_text(_INDEX)
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 100)
    (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"x" * 100)


def _complete_model(model_dir: Path) -> None:
    _write_weights(model_dir)
    (model_dir / "config.json").write_text("{}")


class TestCompletenessRequiresConfig:
    def test_complete_with_config(self, tmp_path: Path) -> None:
        d = tmp_path / "model"
        _complete_model(d)
        assert is_model_directory_complete(d)

    def test_weights_present_but_config_missing_is_incomplete(
        self, tmp_path: Path
    ) -> None:
        # The exact partial-copy shape that crashed the runner: all safetensors
        # + index copied, config.json not yet.
        d = tmp_path / "model"
        _write_weights(d)
        assert not is_model_directory_complete(d)

    def test_missing_a_weight_shard_is_incomplete(self, tmp_path: Path) -> None:
        d = tmp_path / "model"
        _complete_model(d)
        (d / "model-00002-of-00002.safetensors").unlink()
        assert not is_model_directory_complete(d)


class TestShareCompletionStale:
    def test_complete_model_is_not_stale(self, tmp_path: Path) -> None:
        share = tmp_path / "share"
        _complete_model(share / "owner--model")
        assert not share_completion_is_stale(share / "owner--model")

    def test_removed_from_populated_share_is_stale(self, tmp_path: Path) -> None:
        # Model gone, but the share root still holds another model → a genuine
        # removal (reorg / delete), so demote and re-fetch.
        share = tmp_path / "share"
        _complete_model(share / "other--model")
        assert share_completion_is_stale(share / "owner--model")

    def test_partial_copy_on_share_is_stale(self, tmp_path: Path) -> None:
        share = tmp_path / "share"
        _write_weights(share / "owner--model")  # no config.json
        assert share_completion_is_stale(share / "owner--model")

    def test_empty_root_is_not_stale_transient_unmount(self, tmp_path: Path) -> None:
        # Whole share unmounted → empty mountpoint. Must NOT demote every
        # completion, or a transient unmount wipes them all.
        share = tmp_path / "share"
        share.mkdir()
        assert not share_completion_is_stale(share / "owner--model")

    def test_missing_root_is_not_stale(self, tmp_path: Path) -> None:
        assert not share_completion_is_stale(tmp_path / "gone" / "owner--model")
