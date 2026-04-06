# Email Triage Environment - Well Structured Example

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import random

# ============== Pydantic Models ==============


class Email(BaseModel):
    """Represents an incoming email"""

    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    has_attachment: bool = False


class Observation(BaseModel):
    """What the agent sees"""

    current_email: Email
    inbox_count: int = Field(description="Number of emails remaining")
    processed_count: int = Field(description="Emails already processed")
    time_remaining: int = Field(description="Steps remaining in episode")


class Action(BaseModel):
    """What the agent can do"""

    category: Literal["urgent", "important", "normal", "spam", "archive"]
    priority: int = Field(ge=1, le=5, description="1=highest, 5=lowest")
    response_needed: bool = Field(description="Does this need a reply?")
    delegate_to: Optional[str] = Field(
        default=None, description="Team member to delegate to"
    )


class Reward(BaseModel):
    """Feedback signal"""

    score: float = Field(ge=0.0, le=1.0)
    category_correct: bool
    priority_accuracy: float
    time_bonus: float


# ============== Environment ==============


class EmailTriageEnv:
    def __init__(self, task_id: str = "easy"):
        self.task_id = task_id
        self.emails = []
        self.current_idx = 0
        self.processed = []
        self.max_steps = 50
        self.steps = 0

    def reset(self) -> Observation:
        """Reset environment to initial state"""
        self.emails = self._generate_emails()
        self.current_idx = 0
        self.processed = []
        self.steps = 0
        return self._get_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Process one email"""
        email = self.emails[self.current_idx]
        reward = self._calculate_reward(email, action)

        self.processed.append({"email": email, "action": action, "reward": reward})

        self.current_idx += 1
        self.steps += 1

        done = self.current_idx >= len(self.emails) or self.steps >= self.max_steps

        obs = self._get_observation() if not done else self._get_final_observation()

        return obs, reward, done, {"processed": len(self.processed)}

    def state(self) -> dict:
        """Return serializable state"""
        return {
            "task_id": self.task_id,
            "current_idx": self.current_idx,
            "steps": self.steps,
            "processed_count": len(self.processed),
            "emails_remaining": len(self.emails) - self.current_idx,
        }

    def _generate_emails(self) -> List[Email]:
        """Generate emails based on task difficulty"""
        # Implementation varies by task
        templates = self._get_templates_for_task()
        return [Email(**t) for t in templates[:20]]

    def _get_templates_for_task(self) -> List[dict]:
        # Simplified - would have real templates
        return [
            {
                "id": f"email_{i}",
                "sender": f"user{i}@example.com",
                "subject": f"Subject {i}",
                "body": f"Body content {i}",
                "timestamp": "2024-01-01T10:00:00Z",
            }
            for i in range(20)
        ]

    def _calculate_reward(self, email: Email, action: Action) -> Reward:
        """Calculate reward with partial credit"""
        # Ground truth would come from task config
        correct_category = self._get_correct_category(email)
        correct_priority = self._get_correct_priority(email)

        category_correct = action.category == correct_category
        priority_diff = abs(action.priority - correct_priority)
        priority_accuracy = max(0, 1 - priority_diff * 0.25)

        # Time bonus for fast correct decisions
        time_bonus = 0.1 if self.steps < self.max_steps * 0.5 else 0

        base_score = 0.5 if category_correct else 0.1
        score = min(1.0, base_score + priority_accuracy * 0.3 + time_bonus)

        return Reward(
            score=score,
            category_correct=category_correct,
            priority_accuracy=priority_accuracy,
            time_bonus=time_bonus,
        )

    def _get_correct_category(self, email: Email) -> str:
        # Simplified logic
        if "urgent" in email.subject.lower():
            return "urgent"
        return "normal"

    def _get_correct_priority(self, email: Email) -> int:
        return 3  # Simplified

    def _get_observation(self) -> Observation:
        return Observation(
            current_email=self.emails[self.current_idx],
            inbox_count=len(self.emails) - self.current_idx,
            processed_count=len(self.processed),
            time_remaining=self.max_steps - self.steps,
        )

    def _get_final_observation(self) -> Observation:
        return Observation(
            current_email=self.emails[-1],
            inbox_count=0,
            processed_count=len(self.processed),
            time_remaining=0,
        )


# ============== Task Graders ==============


def grade_easy_task(env: EmailTriageEnv) -> float:
    """Easy task: Just categorize emails correctly"""
    if not env.processed:
        return 0.0
    correct = sum(1 for p in env.processed if p["reward"].category_correct)
    return correct / len(env.processed)


def grade_medium_task(env: EmailTriageEnv) -> float:
    """Medium task: Categorize AND prioritize correctly"""
    if not env.processed:
        return 0.0
    total_score = sum(p["reward"].score for p in env.processed)
    return min(1.0, total_score / len(env.processed))


def grade_hard_task(env: EmailTriageEnv) -> float:
    """Hard task: Perfect triage under time pressure with complex emails"""
    if not env.processed:
        return 0.0

    # Penalize if took too many steps
    efficiency = min(1.0, 15 / max(1, len(env.processed)))

    # Average score
    avg_score = sum(p["reward"].score for p in env.processed) / len(env.processed)

    # Bonus for consistency
    scores = [p["reward"].score for p in env.processed]
    variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
    consistency_bonus = 0.1 if variance < 0.05 else 0

    return min(1.0, avg_score * 0.6 + efficiency * 0.3 + consistency_bonus)


TASKS = {
    "easy": {"grader": grade_easy_task, "description": "Categorize emails correctly"},
    "medium": {"grader": grade_medium_task, "description": "Categorize and prioritize"},
    "hard": {"grader": grade_hard_task, "description": "Perfect triage under pressure"},
}
