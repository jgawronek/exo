import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, Self

import psutil

from exo.shared.types.memory import Memory
from exo.shared.types.thunderbolt import ThunderboltIdentifier
from exo.utils.pydantic_ext import FrozenModel


class MemoryUsage(FrozenModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory

    @classmethod
    def from_bytes(
        cls, *, ram_total: int, ram_available: int, swap_total: int, swap_available: int
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
        )

    @classmethod
    def from_psutil(cls, *, override_memory: int | None) -> Self:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        return cls.from_bytes(
            ram_total=vm.total,
            ram_available=vm.available if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
        )

    @classmethod
    def from_cuda(cls, *, override_memory: int | None) -> Self | None:
        """Report a CUDA GPU's VRAM as the node's memory.

        On a discrete NVIDIA GPU the memory that actually bounds MLX inference is
        the GPU's VRAM, not system RAM (unlike Apple Silicon's unified memory).
        Returns None when no GPU/VRAM can be queried so the caller can fall back
        to :meth:`from_psutil`.

        Only a fraction of the free VRAM is advertised as available. Placement
        allocates model weights up to 100% of the advertised memory, but a
        runner additionally needs VRAM for its CUDA context, allocator
        overhead, and warmup KV cache/activations (measured at ~2.7 GB beyond
        the weights on a 24 GB card), so advertising all free VRAM causes
        out-of-memory crashes during warmup.
        """
        vram = _query_cuda_vram_bytes()
        if vram is None:
            return None
        total_vram, free_vram = vram
        usable_vram = int(free_vram * CUDA_VRAM_USABLE_FRACTION)
        sm = psutil.swap_memory()
        return cls.from_bytes(
            ram_total=total_vram,
            ram_available=usable_vram if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
        )


CUDA_VRAM_USABLE_FRACTION: Final = 0.75

# Fraction of Metal's max_recommended_working_set_size that placement may
# fill with model weights. Sustained decode needs headroom within the
# working set for KV cache, activations, and OS pressure. Measured on a
# 256 GB Mac Studio (working set 239e9 bytes): 204e9 bytes of weights ran
# multi-turn generation stably, 211e9 bytes aborted with
# kIOGPUCommandBufferCallbackErrorOutOfMemory. 0.85 advertises 203e9 bytes.
METAL_WORKING_SET_USABLE_FRACTION: Final = 0.85

# Fraction of psutil-available memory advertised on unified-memory CUDA
# nodes (e.g. DGX Spark GB10), where nvidia-smi cannot report VRAM and
# weights are wired into system RAM. Filling 94.7% of available memory with
# weights froze a 128 GB GB10 during model load; 78% ran stably.
CUDA_UNIFIED_USABLE_FRACTION: Final = 0.85


def _query_cuda_vram_bytes() -> tuple[int, int] | None:
    """Total and free VRAM in bytes for the first CUDA GPU via ``nvidia-smi``.

    Returns None if ``nvidia-smi`` is absent or its output cannot be parsed.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None

    lines = completed.stdout.strip().splitlines()
    if not lines:
        return None
    fields = lines[0].split(",")
    if len(fields) != 2:
        return None
    try:
        total_mebibytes = int(fields[0].strip())
        free_mebibytes = int(fields[1].strip())
    except ValueError:
        return None

    mebibyte = 1024 * 1024
    return total_mebibytes * mebibyte, free_mebibytes * mebibyte


class DiskUsage(FrozenModel):
    """Disk space usage for the models directory."""

    total: Memory
    available: Memory

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Get disk usage stats for the partition containing path."""
        total, _used, free = shutil.disk_usage(path)
        return cls(
            total=Memory.from_bytes(total),
            available=Memory.from_bytes(free),
        )


class SystemPerformanceProfile(FrozenModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0


InterfaceType = Literal["wifi", "ethernet", "maybe_ethernet", "thunderbolt", "unknown"]


class NetworkInterfaceInfo(FrozenModel):
    name: str
    ip_address: str
    interface_type: InterfaceType = "unknown"
    # Negotiated link speed in megabits per second (None when the OS cannot
    # report it, e.g. Wi-Fi on some platforms or virtual interfaces).
    link_speed_megabits: int | None = None


class NodeIdentity(FrozenModel):
    """Static and slow-changing node identification data."""

    model_id: str = "Unknown"
    chip_id: str = "Unknown"
    friendly_name: str = "Unknown"
    os_version: str = "Unknown"
    os_build_version: str = "Unknown"


class NodeNetworkInfo(FrozenModel):
    """Network interface information for a node."""

    interfaces: Sequence[NetworkInterfaceInfo] = []


class NodeThunderboltInfo(FrozenModel):
    """Thunderbolt interface identifiers for a node."""

    interfaces: Sequence[ThunderboltIdentifier] = []


class NodeRdmaCtlStatus(FrozenModel):
    """Whether RDMA is enabled on this node (via rdma_ctl)."""

    enabled: bool


class ThunderboltBridgeStatus(FrozenModel):
    """Whether the Thunderbolt Bridge network service is enabled on this node."""

    enabled: bool
    exists: bool
    service_name: str | None = None
