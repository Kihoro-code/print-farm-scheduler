# Missing Dockerfile Edge Case - Data Cleaning Environment

from pydantic import BaseModel, Field
from typing import List, Optional


class DataRow(BaseModel):
    id: int
    values: List[Optional[float]]
    has_nulls: bool
    has_outliers: bool


class Observation(BaseModel):
    current_row: DataRow
    rows_remaining: int
    cleaned_count: int


class Action(BaseModel):
    operation: str  # "keep", "drop", "impute_mean", "impute_median", "cap_outliers"
    column_indices: List[int] = Field(default_factory=list)


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    data_quality_improvement: float


class DataCleaningEnv:
    def __init__(self, task_id: str = "easy"):
        self.task_id = task_id
        self.data = []
        self.idx = 0
        self.cleaned = []

    def reset(self) -> Observation:
        self.data = [
            DataRow(id=i, values=[1.0, None, 3.0], has_nulls=True, has_outliers=False)
            for i in range(10)
        ]
        self.idx = 0
        self.cleaned = []
        return self._get_obs()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        row = self.data[self.idx]
        # Simplified scoring
        score = (
            0.8
            if action.operation in ["impute_mean", "impute_median"] and row.has_nulls
            else 0.3
        )
        self.cleaned.append((row, action, score))
        self.idx += 1
        done = self.idx >= len(self.data)
        return (
            self._get_obs(),
            Reward(score=score, data_quality_improvement=score * 0.1),
            done,
            {},
        )

    def state(self) -> dict:
        return {"idx": self.idx, "cleaned": len(self.cleaned)}

    def _get_obs(self) -> Observation:
        return Observation(
            current_row=self.data[min(self.idx, len(self.data) - 1)],
            rows_remaining=len(self.data) - self.idx,
            cleaned_count=len(self.cleaned),
        )


def grade_easy(env):
    return sum(c[2] for c in env.cleaned) / max(1, len(env.cleaned))


def grade_medium(env):
    return grade_easy(env) * 0.9


def grade_hard(env):
    return grade_easy(env) * 0.7


TASKS = {
    "easy": {"grader": grade_easy},
    "medium": {"grader": grade_medium},
    "hard": {"grader": grade_hard},
}
