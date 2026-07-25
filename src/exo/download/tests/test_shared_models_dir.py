"""Tests for the runtime shared models directory: validation and path preference."""

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from exo.download.download_utils import (
    build_model_path,
    resolve_existing_model,
    select_download_dir,
)
from exo.download.shared_models_dir import (
    get_shared_models_dir,
    set_shared_models_dir,
    validate_shared_models_directory,
)
from exo.shared.types.common import ModelId

MODEL_ID = ModelId("test-org/test-model")
NORMALIZED = MODEL_ID.normalize()


def _create_complete_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {"layer.weight": "model.safetensors"}
    index = {"metadata": {"total_size": 1024}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "config.json").write_text('{"model_type": "test"}')


@pytest.fixture(autouse=True)
def reset_shared_models_dir() -> Iterator[None]:
    previous = get_shared_models_dir()
    yield
    set_shared_models_dir(previous)


class TestValidateSharedModelsDirectory:
    def test_missing_path_is_invalid(self, tmp_path: Path) -> None:
        status = validate_shared_models_directory(str(tmp_path / "missing"))
        assert not status.valid
        assert status.error == "Path does not exist"

    def test_file_is_invalid(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file"
        file_path.write_text("not a directory")
        status = validate_shared_models_directory(str(file_path))
        assert not status.valid
        assert status.error == "Path is not a directory"

    def test_writable_directory_is_valid(self, tmp_path: Path) -> None:
        status = validate_shared_models_directory(str(tmp_path))
        assert status.valid
        assert status.error is None
        assert status.free_bytes is not None and status.free_bytes > 0

    def test_leaves_no_probe_files_behind(self, tmp_path: Path) -> None:
        validate_shared_models_directory(str(tmp_path))
        assert list(tmp_path.iterdir()) == []


class TestSharedDirPreference:
    def test_resolve_prefers_shared_dir(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        _create_complete_model(shared / NORMALIZED)
        writable = tmp_path / "writable"
        _create_complete_model(writable / NORMALIZED)
        set_shared_models_dir(shared)
        with (
            patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
        ):
            assert resolve_existing_model(MODEL_ID) == shared / NORMALIZED

    def test_resolve_falls_back_to_local_dirs(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        shared.mkdir()
        writable = tmp_path / "writable"
        _create_complete_model(writable / NORMALIZED)
        set_shared_models_dir(shared)
        with (
            patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
        ):
            assert resolve_existing_model(MODEL_ID) == writable / NORMALIZED

    def test_build_model_path_targets_shared_dir_for_new_models(
        self, tmp_path: Path
    ) -> None:
        shared = tmp_path / "shared"
        shared.mkdir()
        writable = tmp_path / "writable"
        writable.mkdir()
        set_shared_models_dir(shared)
        with (
            patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
            patch("exo.download.download_utils.EXO_DEFAULT_MODELS_DIR", writable),
        ):
            assert build_model_path(MODEL_ID) == shared / NORMALIZED

    def test_build_model_path_uses_default_without_shared_dir(
        self, tmp_path: Path
    ) -> None:
        writable = tmp_path / "writable"
        writable.mkdir()
        set_shared_models_dir(None)
        with (
            patch("exo.download.download_utils.EXO_MODELS_READ_ONLY_DIRS", ()),
            patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)),
            patch("exo.download.download_utils.EXO_DEFAULT_MODELS_DIR", writable),
        ):
            assert build_model_path(MODEL_ID) == writable / NORMALIZED

    def test_select_download_dir_prefers_shared_dir(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        shared.mkdir()
        writable = tmp_path / "writable"
        writable.mkdir()
        set_shared_models_dir(shared)
        with patch("exo.download.download_utils.EXO_MODELS_DIRS", (writable,)):
            assert select_download_dir(required_bytes=1) == shared
