import asyncio
import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Generator, Iterable, Literal

import aiofiles
import aiofiles.os as aios
from loguru import logger

from exo.shared.types.worker.shards import ShardMetadata

# Where the active Hugging Face token comes from. "env" wins over "file".
TokenSource = Literal["env", "file", "none"]


def filter_repo_objects[T](
    items: Iterable[T],
    *,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    key: Callable[[T], str] | None = None,
) -> Generator[T, None, None]:
    if isinstance(allow_patterns, str):
        allow_patterns = [allow_patterns]
    if isinstance(ignore_patterns, str):
        ignore_patterns = [ignore_patterns]
    if allow_patterns is not None:
        allow_patterns = [_add_wildcard_to_directories(p) for p in allow_patterns]
    if ignore_patterns is not None:
        ignore_patterns = [_add_wildcard_to_directories(p) for p in ignore_patterns]

    if key is None:

        def _identity(item: T) -> str:
            if isinstance(item, str):
                return item
            if isinstance(item, Path):
                return str(item)
            raise ValueError(
                f"Please provide `key` argument in `filter_repo_objects`: `{item}` is not a string."
            )

        key = _identity

    for item in items:
        path = key(item)
        if allow_patterns is not None and not any(
            fnmatch(path, r) for r in allow_patterns
        ):
            continue
        if ignore_patterns is not None and any(
            fnmatch(path, r) for r in ignore_patterns
        ):
            continue
        yield item


def _add_wildcard_to_directories(pattern: str) -> str:
    if pattern[-1] == "/":
        return pattern + "*"
    return pattern


def get_hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def get_hf_home() -> Path:
    """Get the Hugging Face home directory."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


async def get_hf_token() -> str | None:
    """Retrieve the Hugging Face token from HF_TOKEN env var or HF_HOME directory."""
    # Check environment variable first
    if token := os.environ.get("HF_TOKEN"):
        return token
    # Fall back to file-based token
    token_path = get_hf_home() / "token"
    if await aios.path.exists(token_path):
        async with aiofiles.open(token_path, "r") as f:
            return (await f.read()).strip()
    return None


async def get_auth_headers() -> dict[str, str]:
    """Get authentication headers if a token is available."""
    token = await get_hf_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def get_hf_token_path() -> Path:
    """Location of the file-based token, shared with the `hf` CLI."""
    return get_hf_home() / "token"


async def get_hf_token_source() -> TokenSource:
    """Where the active token comes from, without revealing it.

    ``HF_TOKEN`` shadows the file in :func:`get_hf_token`, so a token written
    from the dashboard has no effect while the env var is set. Callers surface
    this so the UI can say so rather than silently ignoring the saved token.
    """
    if os.environ.get("HF_TOKEN"):
        return "env"
    if await aios.path.exists(get_hf_token_path()):
        return "file"
    return "none"


def mask_hf_token(token: str) -> str:
    """Render a token as a non-recoverable hint, e.g. ``hf_ab…7f9c``."""
    if len(token) <= 8:
        return "…"
    return f"{token[:5]}…{token[-4:]}"


def _write_token_file(token_path: Path, token: str) -> None:
    """Write the token 0600, never leaving it briefly world-readable.

    O_CREAT's mode only applies when the file is created, so an existing file
    with laxer permissions is tightened explicitly afterwards.
    """
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        _ = os.write(fd, token.encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(token_path, 0o600)


async def set_hf_token(token: str) -> None:
    """Persist a token to the shared HF location with owner-only permissions."""
    token_path = get_hf_token_path()
    await aios.makedirs(token_path.parent, exist_ok=True)
    await asyncio.to_thread(_write_token_file, token_path, token.strip())
    logger.info(f"Stored Hugging Face token at {token_path}")


async def delete_hf_token() -> bool:
    """Remove the file-based token. Returns whether a file was removed."""
    token_path = get_hf_token_path()
    if not await aios.path.exists(token_path):
        return False
    await aios.remove(token_path)
    logger.info(f"Removed Hugging Face token at {token_path}")
    return True


def extract_layer_num(tensor_name: str) -> int | None:
    # This is a simple example and might need to be adjusted based on the actual naming convention
    parts = tensor_name.split(".")
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def get_allow_patterns(weight_map: dict[str, str], shard: ShardMetadata) -> list[str]:
    default_patterns = set(
        [
            "*.json",
            "*.py",
            "tokenizer.model",
            "tiktoken.model",
            "*/spiece.model",
            "*.tiktoken",
            "*.txt",
            "*.jinja",
        ]
    )
    shard_specific_patterns: set[str] = set()

    if shard.model_card.components is not None:
        shardable_component = next(
            (c for c in shard.model_card.components if c.can_shard), None
        )

        if weight_map and shardable_component:
            for tensor_name, filename in weight_map.items():
                # Strip component prefix from tensor name (added by weight map namespacing)
                # E.g., "transformer/blocks.0.weight" -> "blocks.0.weight"
                if "/" in tensor_name:
                    _, tensor_name_no_prefix = tensor_name.split("/", 1)
                else:
                    tensor_name_no_prefix = tensor_name

                # Determine which component this file belongs to from filename
                component_path = Path(filename).parts[0] if "/" in filename else None

                if component_path == shardable_component.component_path.rstrip("/"):
                    layer_num = extract_layer_num(tensor_name_no_prefix)
                    if (
                        layer_num is not None
                        and shard.start_layer <= layer_num < shard.end_layer
                    ):
                        shard_specific_patterns.add(filename)

                    if shard.is_first_layer or shard.is_last_layer:
                        shard_specific_patterns.add(filename)
                else:
                    shard_specific_patterns.add(filename)

        else:
            shard_specific_patterns = set(["*.safetensors"])

        # TODO(ciaran): temporary - Include all files from non-shardable components that have no index file
        for component in shard.model_card.components:
            if not component.can_shard and component.safetensors_index_filename is None:
                component_pattern = f"{component.component_path.rstrip('/')}/*"
                shard_specific_patterns.add(component_pattern)
    else:
        if weight_map:
            for tensor_name, filename in weight_map.items():
                layer_num = extract_layer_num(tensor_name)
                if (
                    layer_num is not None
                    and shard.start_layer <= layer_num < shard.end_layer
                ):
                    shard_specific_patterns.add(filename)
            layer_independent_files = set(
                [v for k, v in weight_map.items() if extract_layer_num(k) is None]
            )
            shard_specific_patterns.update(layer_independent_files)
            logger.debug(f"get_allow_patterns {shard=} {layer_independent_files=}")
        else:
            shard_specific_patterns = set(["*.safetensors"])

    logger.info(f"get_allow_patterns {shard=} {shard_specific_patterns=}")
    return list(default_patterns | shard_specific_patterns)
