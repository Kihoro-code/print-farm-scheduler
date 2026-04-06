# Customer Support Ticket Router - Only 2 Tasks (Edge Case)

from pydantic import BaseModel, Field
from typing import Literal


class Ticket(BaseModel):
    id: str
    customer_id: str
    subject: str
    body: str
    category: str


class Observation(BaseModel):
    current_ticket: Ticket
    queue_length: int


class Action(BaseModel):
    route_to: Literal["billing", "technical", "general", "escalate"]
    priority: int = Field(ge=1, le=3)


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)


class TicketRouterEnv:
    def __init__(self, task_id: str = "easy"):
        self.task_id = task_id
        self.tickets = []
        self.idx = 0

    def reset(self) -> Observation:
        self.tickets = [
            Ticket(
                id="1",
                customer_id="c1",
                subject="Billing issue",
                body="Can't pay",
                category="billing",
            ),
            Ticket(
                id="2",
                customer_id="c2",
                subject="App crash",
                body="Error 500",
                category="technical",
            ),
        ]
        self.idx = 0
        return Observation(current_ticket=self.tickets[0], queue_length=2)

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        correct = self.tickets[self.idx].category == action.route_to
        self.idx += 1
        done = self.idx >= len(self.tickets)
        obs = Observation(
            current_ticket=self.tickets[min(self.idx, len(self.tickets) - 1)],
            queue_length=len(self.tickets) - self.idx,
        )
        return obs, Reward(score=1.0 if correct else 0.0), done, {}

    def state(self) -> dict:
        return {"idx": self.idx, "task_id": self.task_id}


# ONLY 2 TASKS - This should fail V-5 validation
def grade_easy(env):
    return 0.8


def grade_medium(env):
    return 0.6


TASKS = {
    "easy": {"grader": grade_easy, "description": "Route tickets correctly"},
    "medium": {"grader": grade_medium, "description": "Route with priority"},
    # Missing hard task!
}
