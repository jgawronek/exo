import tomllib
from pathlib import Path
from typing import Final, cast

LINUX_CUDA_13_TORCH_REQUIREMENTS: Final = frozenset(
    {
        "torch==2.10.0; sys_platform == 'linux'",
        "torchaudio==2.10.0; sys_platform == 'linux'",
        "torchvision==0.25.0; sys_platform == 'linux'",
    }
)
LINUX_CUDA_13_TORCH_VERSIONS: Final = (
    ("torch", "2.10.0+cu130"),
    ("torchaudio", "2.10.0+cu130"),
    ("torchvision", "0.25.0+cu130"),
)
PYTORCH_CUDA_13_INDEX: Final = "https://download.pytorch.org/whl/cu130"


def _load_toml(path: Path) -> dict[str, object]:
    return cast(dict[str, object], tomllib.loads(path.read_text()))


def test_mlx_cuda13_extra_resolves_linux_torch_dependencies() -> None:
    repository_root = Path(__file__).parents[4]
    project_configuration = _load_toml(repository_root / "pyproject.toml")
    project_metadata = cast(dict[str, object], project_configuration["project"])
    optional_dependencies = cast(
        dict[str, list[str]],
        project_metadata["optional-dependencies"],
    )
    assert set(optional_dependencies["mlx-cuda13"]) >= LINUX_CUDA_13_TORCH_REQUIREMENTS

    lock = _load_toml(repository_root / "uv.lock")
    packages = cast(list[dict[str, object]], lock["package"])
    exo_package = next(package for package in packages if package["name"] == "exo")
    locked_optional_dependencies = cast(
        dict[str, list[dict[str, object]]],
        exo_package["optional-dependencies"],
    )
    cuda_13_dependencies = locked_optional_dependencies["mlx-cuda13"]

    for package_name, expected_version in LINUX_CUDA_13_TORCH_VERSIONS:
        dependency = next(
            dependency
            for dependency in cuda_13_dependencies
            if dependency["name"] == package_name
            and dependency.get("version") == expected_version
        )
        source = cast(dict[str, str], dependency["source"])
        marker = cast(str, dependency["marker"])
        assert source["registry"] == PYTORCH_CUDA_13_INDEX
        assert "sys_platform == 'linux'" in marker
        assert "extra == 'extra-3-exo-mlx-cuda13'" in marker
