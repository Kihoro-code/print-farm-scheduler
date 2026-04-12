"""Data models for the Print Farm Scheduler environment."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from openenv.core.env_server.types import Action, Observation


class MachineSnapshot(BaseModel):
    """Snapshot of a single machine's current state."""
    id: int
    status: Literal["idle", "printing", "changing_spool"]
    material_loaded: str
    filament_remaining_g: float
    speed_modifier: float = 1.0
    max_weight_g: float = 100.0
    hours_used: float = 0.0
    current_job_id: Optional[int] = None
    job_progress_steps: int = 0
    job_total_steps: int = 0


class JobSnapshot(BaseModel):
    """Snapshot of a single job's current state."""
    id: int
    material: str
    weight_g: float
    print_steps: int
    deadline_step: int
    assigned_machine: Optional[int] = None


class PrintFarmAction(Action):
    """
    Action for the Print Farm Scheduler environment.

    action_type values:
    - "assign": assign first job in queue to an idle machine
    - "preempt": cancel a running job on a machine
    - "prioritize": move a specific job to the front of the queue
    - "skip": do nothing this step
    """
    type: Literal["assign", "preempt", "prioritize", "skip"] = Field(
        ..., description="Type of scheduling action to perform"
    )
    machine_id: Optional[int] = Field(
        None, description="Target machine ID (for assign/preempt)"
    )
    job_id: Optional[int] = Field(
        None, description="Target job ID (for prioritize)"
    )


class RewardBreakdown(BaseModel):
    """Detailed reward breakdown for interpretability."""
    value: float = Field(description="Total reward value in (0.0, 1.0)")
    breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Component-wise reward breakdown"
    )
    reason: str = Field(default="", description="Human-readable reason for reward")


class PrintFarmObservation(Observation):
    """
    Observation returned by the Print Farm Scheduler environment.

    Contains full factory state: machines, job queue, pending arrivals,
    and episode statistics.
    """
    step: int = Field(0, description="Current step number in the episode")
    machines: List[MachineSnapshot] = Field(
        default_factory=list, description="Current state of all machines"
    )
    queue: List[JobSnapshot] = Field(
        default_factory=list, description="Jobs waiting to be assigned"
    )
    pending_arrivals: List[JobSnapshot] = Field(
        default_factory=list,
        description="Advance customer bookings visible ≤2 steps ahead",
    )
    completed_count: int = Field(0, description="Jobs completed so far")
    deadlines_missed: int = Field(0, description="Deadlines missed so far")
    total_jobs_ever: int = Field(0, description="Total jobs seen this episode")
    reward_info: Optional[RewardBreakdown] = Field(
        None, description="Reward breakdown from last step"
    )
    rubric_score: Optional[float] = Field(
        None, description="Score produced by the OpenEnv rubric layer"
    )


__all__ = [
    "MachineSnapshot",
    "JobSnapshot",
    "PrintFarmAction",
    "PrintFarmObservation",
    "RewardBreakdown",
]
