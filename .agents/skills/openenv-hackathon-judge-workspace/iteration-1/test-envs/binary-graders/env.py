# Binary Graders Edge Case - Content Moderation Environment
# Problem: Graders only return 0.0 or 1.0, no partial credit

from pydantic import BaseModel, Field
from typing import Literal


class Post(BaseModel):
    id: str
    content: str
    author: str
    reported: bool


class Observation(BaseModel):
    current_post: Post
    queue_size: int


class Action(BaseModel):
    decision: Literal["approve", "remove", "flag_review"]
    reason: str = ""


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)


class ModerationEnv:
    def __init__(self, task_id: str = "easy"):
        self.task_id = task_id
        self.posts = []
        self.idx = 0
        self.decisions = []

    def reset(self) -> Observation:
        self.posts = [
            Post(id="1", content="Hello world", author="user1", reported=False),
            Post(id="2", content="Buy cheap stuff", author="spammer", reported=True),
            Post(id="3", content="Great product!", author="user2", reported=False),
        ]
        self.idx = 0
        self.decisions = []
        return Observation(current_post=self.posts[0], queue_size=3)

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        post = self.posts[self.idx]
        # Binary reward - PROBLEM: no partial credit
        correct = (post.reported and action.decision == "remove") or (
            not post.reported and action.decision == "approve"
        )
        score = 1.0 if correct else 0.0  # <-- This is the problem!

        self.decisions.append({"post": post, "action": action, "correct": correct})
        self.idx += 1
        done = self.idx >= len(self.posts)

        obs = Observation(
            current_post=self.posts[min(self.idx, len(self.posts) - 1)],
            queue_size=len(self.posts) - self.idx,
        )
        return obs, Reward(score=score), done, {}

    def state(self) -> dict:
        return {"idx": self.idx, "decisions": len(self.decisions)}


# PROBLEM: These graders only return 0.0 or 1.0 based on binary conditions
def grade_easy(env):
    if not env.decisions:
        return 0.0
    correct = sum(1 for d in env.decisions if d["correct"])
    # Returns 1.0 if all correct, 0.0 otherwise - no partial credit!
    return 1.0 if correct == len(env.decisions) else 0.0


def grade_medium(env):
    if not env.decisions:
        return 0.0
    correct = sum(1 for d in env.decisions if d["correct"])
    # Same problem - binary output
    return 1.0 if correct >= len(env.decisions) * 0.8 else 0.0


def grade_hard(env):
    if not env.decisions:
        return 0.0
    correct = sum(1 for d in env.decisions if d["correct"])
    # Binary: either perfect or zero
    return 1.0 if correct == len(env.decisions) else 0.0


TASKS = {
    "easy": {"grader": grade_easy, "description": "Basic moderation"},
    "medium": {"grader": grade_medium, "description": "80% accuracy threshold"},
    "hard": {"grader": grade_hard, "description": "Perfect accuracy required"},
}
