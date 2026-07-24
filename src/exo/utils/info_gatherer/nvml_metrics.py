from contextlib import suppress
from typing import Self

from loguru import logger

from exo.shared.types.profiling import (
    SystemPerformanceProfile,
    pinned_cuda_device_index,
)
from exo.utils.pydantic_ext import TaggedModel


class NvmlMetrics(TaggedModel):
    """GPU load/temp/power sampled via NVML for CUDA nodes."""

    system_profile: SystemPerformanceProfile

    @classmethod
    def gather(cls) -> Self | None:
        try:
            import pynvml as nvml  # pyright: ignore[reportMissingModuleSource]
        except ImportError:
            return None

        try:
            nvml.nvmlInit()
        except Exception:
            return None

        try:
            pinned = pinned_cuda_device_index()
            device_index = int(pinned) if pinned is not None else 0
            device_count = nvml.nvmlDeviceGetCount()
            if device_count <= 0:
                return None
            if device_index >= device_count:
                device_index = 0

            handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = nvml.nvmlDeviceGetTemperature(
                handle, nvml.NVML_TEMPERATURE_GPU
            )

            power_watts = 0.0
            with suppress(Exception):
                # Not all devices expose power readings
                power_watts = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

            return cls(
                system_profile=SystemPerformanceProfile(
                    gpu_usage=float(utilization.gpu) / 100.0,
                    temp=float(temperature),
                    sys_power=power_watts,
                )
            )
        except Exception as error:
            logger.opt(exception=error).debug("NVML metrics gather failed")
            return None
        finally:
            with suppress(Exception):
                nvml.nvmlShutdown()
