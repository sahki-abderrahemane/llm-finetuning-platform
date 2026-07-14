"""
Hardware detection utilities.

Automatically detects the best execution device and precision
for the current machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class DeviceInfo:
    """
    Information about the detected execution device.
    """

    device: str
    use_cpu: bool

    fp16: bool
    bf16: bool

    gpu_name: str | None


class DeviceDetector:
    """
    Detects the best available hardware configuration.
    """

    @staticmethod
    def detect() -> DeviceInfo:
        """
        Detect the execution device and supported precision.
        """

        if not torch.cuda.is_available():
            return DeviceInfo(
                device="cpu",
                use_cpu=True,
                fp16=False,
                bf16=False,
                gpu_name=None,
            )

        bf16_supported = torch.cuda.is_bf16_supported()

        return DeviceInfo(
            device="cuda",
            use_cpu=False,
            fp16=not bf16_supported,
            bf16=bf16_supported,
            gpu_name=torch.cuda.get_device_name(0),
        )

    @staticmethod
    def print_summary(device: DeviceInfo) -> None:
        """
        Print a summary of the detected hardware.
        """

        print("=" * 60)
        print("Hardware Summary")
        print("=" * 60)
        print(f"Device      : {device.device}")
        print(f"GPU         : {device.gpu_name or 'N/A'}")
        print(f"FP16        : {device.fp16}")
        print(f"BF16        : {device.bf16}")
        print(f"PyTorch     : {torch.__version__}")

        if torch.cuda.is_available():
            print(f"CUDA        : {torch.version.cuda}")

        print("=" * 60)