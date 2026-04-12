from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from simulator import PrintFarmSimulator


class MachineSnapshot(BaseModel):
    id: int
    status: Literal["idle", "printing", "changing_spool"]
    material_loaded: str
    filament_remaining_g: float
    speed_modifier: float = 1.0
    max_weight_g: float = 100.0
    hours_used: float = 0.0
    current_job_id: int | None = None
    job_progress_steps: int = 0
    job_total_steps: int = 0


class JobSnapshot(BaseModel):
    id: int
    material: str
    weight_g: float
    print_steps: int
    deadline_step: int
    assigned_machine: int | None = None


class Observation(BaseModel):
    step: int
    machines: list[MachineSnapshot]
    queue: list[JobSnapshot]
    pending_arrivals: list[JobSnapshot]
    completed_count: int
    deadlines_missed: int
    total_jobs_ever: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    type: Literal["assign", "preempt", "prioritize", "skip"]
    machine_id: int | None = None
    job_id: int | None = None


class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    breakdown: dict[str, float]
    reason: str


class PrintFarmEnv:
    """
    Standalone wrapper used by inference.py.

    The simulation rules live in PrintFarmSimulator so this wrapper and the
    OpenEnv server stay in sync.
    """

    STEP_DURATION_MINUTES = PrintFarmSimulator.STEP_DURATION_MINUTES

    def __init__(
        self,
        seed: int | None = None,
        difficulty: str = "easy",
        max_steps: int = 30,
        **config: Any,
    ) -> None:
        self._simulator = PrintFarmSimulator(
            seed=seed,
            difficulty=difficulty,
            max_steps=max_steps,
            config=config,
        )
        self.MAX_STEPS = self._simulator.max_steps

    @property
    def _step_count(self) -> int:
        return self._simulator.step_count

    def reset(
        self,
        *,
        seed: int | None = None,
        difficulty: str | None = None,
        max_steps: int | None = None,
        **config: Any,
    ) -> Observation:
        payload = self._simulator.reset(
            seed=seed,
            difficulty=difficulty,
            max_steps=max_steps,
            **config,
        )
        self.MAX_STEPS = self._simulator.max_steps
        return self._parse_observation(payload)

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        payload, reward_payload, done = self._simulator.step(
            action.model_dump(exclude_none=True)
        )
        observation = self._parse_observation(payload)
        reward = Reward(**reward_payload)
        info = {
            "step": self._simulator.step_count,
            "env_id": "print-farm-scheduler",
            "difficulty": self._simulator.difficulty,
        }
        return observation, reward, done, info

    def state(self) -> dict[str, Any]:
        return self._simulator.state_snapshot()

    def _is_valid_action(self, action: Action) -> bool:
        return self._simulator.is_valid_action(action.model_dump(exclude_none=True))

    def _parse_observation(self, payload: dict[str, Any]) -> Observation:
        return Observation(
            step=payload["step"],
            machines=[MachineSnapshot(**machine) for machine in payload["machines"]],
            queue=[JobSnapshot(**job) for job in payload["queue"]],
            pending_arrivals=[
                JobSnapshot(**job) for job in payload["pending_arrivals"]
            ],
            completed_count=payload["completed_count"],
            deadlines_missed=payload["deadlines_missed"],
            total_jobs_ever=payload["total_jobs_ever"],
            metadata=payload.get("metadata", {}),
        )
