"""
Print Farm Scheduler OpenEnv environment wrapper.

The simulation rules live in the shared PrintFarmSimulator so the standalone
inference env and the server env stay behaviorally aligned.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import (
        JobSnapshot,
        MachineSnapshot,
        PrintFarmAction,
        PrintFarmObservation,
        RewardBreakdown,
    )
    from simulator import PrintFarmSimulator
    from server.rubrics import PrintFarmCompositeRubric
except ImportError:
    from ..models import (
        JobSnapshot,
        MachineSnapshot,
        PrintFarmAction,
        PrintFarmObservation,
        RewardBreakdown,
    )
    from ..simulator import PrintFarmSimulator
    from .rubrics import PrintFarmCompositeRubric


class PrintFarmEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    STEP_DURATION_MINUTES: int = PrintFarmSimulator.STEP_DURATION_MINUTES

    def __init__(self) -> None:
        self._simulator = PrintFarmSimulator(seed=42, difficulty="easy", max_steps=30)
        self._state_obj = State(episode_id=str(uuid4()), step_count=0)
        self._rubric = PrintFarmCompositeRubric(difficulty=self._simulator.difficulty)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        max_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> PrintFarmObservation:
        payload = self._simulator.reset(
            seed=seed,
            difficulty=difficulty,
            max_steps=max_steps,
            **kwargs,
        )
        self._state_obj = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._rubric.reset()
        self._rubric.set_difficulty(self._simulator.difficulty)
        return self._parse_observation(payload, done=False, reward=None, reward_info=None)

    def step(
        self,
        action: PrintFarmAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PrintFarmObservation:
        if not self._simulator.is_valid_action(action.model_dump(exclude_none=True)):
            raise ValueError(f"Illegal action: {action}")

        payload, reward_payload, done = self._simulator.step(
            action.model_dump(exclude_none=True)
        )
        self._state_obj.step_count = self._simulator.step_count

        reward_info = RewardBreakdown(**reward_payload)
        observation = self._parse_observation(
            payload,
            done=done,
            reward=reward_info.value,
            reward_info=reward_info,
        )
        observation.rubric_score = float(self._rubric(action, observation))
        return observation

    @property
    def state(self) -> State:
        return self._state_obj

    def _parse_observation(
        self,
        payload: dict[str, Any],
        *,
        done: bool,
        reward: Optional[float],
        reward_info: Optional[RewardBreakdown],
    ) -> PrintFarmObservation:
        return PrintFarmObservation(
            step=payload["step"],
            machines=[MachineSnapshot(**machine) for machine in payload["machines"]],
            queue=[JobSnapshot(**job) for job in payload["queue"]],
            pending_arrivals=[
                JobSnapshot(**job) for job in payload["pending_arrivals"]
            ],
            completed_count=payload["completed_count"],
            deadlines_missed=payload["deadlines_missed"],
            total_jobs_ever=payload["total_jobs_ever"],
            done=done,
            reward=reward,
            metadata=payload.get("metadata", {}),
            reward_info=reward_info,
        )
