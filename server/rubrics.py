"""Rubrics for the Print Farm Scheduler environment."""

from __future__ import annotations

from typing import Any

from openenv.core.rubrics.base import Rubric


class PrintFarmOutcomeRubric(Rubric):
    """Return the task-aligned score at the end of an episode."""

    def __init__(self, difficulty: str = "easy") -> None:
        super().__init__()
        self._difficulty = difficulty

    def set_difficulty(self, difficulty: str) -> None:
        self._difficulty = difficulty

    def forward(self, action: Any, observation: Any) -> float:
        if not getattr(observation, "done", False):
            return 0.0
        metadata = getattr(observation, "metadata", {}) or {}
        return float(
            metadata.get(
                "active_difficulty_score",
                metadata.get("score_estimates", {}).get(self._difficulty, 0.0),
            )
        )


class PrintFarmProcessRubric(Rubric):
    """Use dense per-step progress as the process signal."""

    def forward(self, action: Any, observation: Any) -> float:
        metadata = getattr(observation, "metadata", {}) or {}
        progress_signal = metadata.get("progress_signal")
        if progress_signal is not None:
            return float(progress_signal)
        reward = getattr(observation, "reward", None)
        return float(reward) if reward is not None else 0.0


class PrintFarmCompositeRubric(Rubric):
    """Process reward during an episode, task score at termination."""

    def __init__(
        self,
        difficulty: str = "easy",
        outcome: Rubric | None = None,
        process: Rubric | None = None,
    ) -> None:
        super().__init__()
        self.outcome = outcome or PrintFarmOutcomeRubric(difficulty=difficulty)
        self.process = process or PrintFarmProcessRubric()

    def set_difficulty(self, difficulty: str) -> None:
        if hasattr(self.outcome, "set_difficulty"):
            self.outcome.set_difficulty(difficulty)

    def forward(self, action: Any, observation: Any) -> float:
        if getattr(observation, "done", False):
            return self.outcome(action, observation)
        return self.process(action, observation)

    def reset(self) -> None:
        if hasattr(self.outcome, "reset"):
            self.outcome.reset()
        if hasattr(self.process, "reset"):
            self.process.reset()
