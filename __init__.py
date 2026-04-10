"""Print Farm Scheduler Environment package exports."""

from typing import Any

try:
    from .models import (
        MachineSnapshot,
        JobSnapshot,
        PrintFarmAction,
        PrintFarmObservation,
        RewardBreakdown,
    )
except ImportError:
    from models import (
        MachineSnapshot,
        JobSnapshot,
        PrintFarmAction,
        PrintFarmObservation,
        RewardBreakdown,
    )

__all__ = [
    "MachineSnapshot",
    "JobSnapshot",
    "PrintFarmAction",
    "PrintFarmObservation",
    "RewardBreakdown",
    "PrintFarmEnv",
]


def __getattr__(name: str) -> Any:
    if name == "PrintFarmEnv":
        try:
            from .client import PrintFarmEnv as _PrintFarmEnv
        except ImportError:
            from client import PrintFarmEnv as _PrintFarmEnv
        return _PrintFarmEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
