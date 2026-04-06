# OpenEnv Official Evaluation: 37/100

## Pre-Submission Validation

| Check | Status | Notes |
|-------|--------|-------|
| V-1: HF Space deploys | ❌ | No HF Space config in README (README.md missing). No `app.py` or `main.py` with Gradio/FastAPI endpoints found. |
| V-2: OpenEnv spec compliance | ✅ | `openenv.yaml` present with required fields. `step()`, `reset()`, `state()` methods implemented in `env.py` with correct signatures. |
| V-3: Dockerfile builds | ✅ | Dockerfile exists with valid base image (`python:3.11-slim`), pip installs, and entry point. Note: references `app:app` but no `app.py` exists. |
| V-4: Baseline reproduces | ❌ | No baseline inference script found. No file reads `OPENAI_API_KEY` or calls OpenAI-compatible API. |
| V-5: 3+ tasks with graders | ❌ | **CRITICAL FAILURE**: Only 2 tasks implemented (`easy`, `medium`). Missing required 3rd task. |

### Required Endpoints
| Endpoint | Status | Notes |
|----------|--------|-------|
| `/baseline` | ❌ | No route definition found. No `app.py` or API file exists. |
| `/grader` | ❌ | No route definition found. Grader functions exist but not exposed via API. |
| `/tasks` | ❌ | No route definition found. `TASKS` dict exists in `env.py` but not exposed via API. |

### Disqualification Gate
- DQ-1 (deploys): ✅ — Dockerfile exists and `openenv.yaml` present with core methods implemented.
- DQ-2 (original): ✅ — Novel customer support ticket router implementation, not a copy of standard Gym environments.
- DQ-3 (graders vary): 🚫 — **FAIL**: Graders return constant values (`grade_easy` always returns 0.8, `grade_medium` always returns 0.6). No conditional logic based on env state.
- DQ-4 (baseline script): 🚫 — **FAIL**: No baseline inference script exists.

> **Disqualified: YES** (DQ-3 and DQ-4 triggered, V-5 validation failed)

---

## Scoring Breakdown

### 1. Real-World Utility: 18/30 — Band C
**Domain**: Customer support ticket routing
**Assessment**: Customer support ticket routing is a legitimate real-world task that knowledge workers perform. The domain selection is good — routing tickets to billing, technical, general, or escalation queues is something humans do. However, the implementation is shallow: only 2 trivially simple tickets with obvious categorization. The inputs are not realistic (extremely short body text like "Can't pay" and "Error 500"). Would benefit from realistic email content and nuanced routing decisions.

### 2. Task & Grader Quality: 4/25
| Check | Points | Evidence |
|-------|--------|----------|
| 3+ tasks | 0/7 | **CRITICAL**: Only 2 tasks found (`easy`, `medium`). Required minimum is 3. See `env.py:78-81` and `openenv.yaml:13-17`. |
| Grader range [0.0–1.0] | 4/6 | Graders return values in valid range (0.8 and 0.6). `Reward` model has `Field(ge=0.0, le=1.0)` constraint. |
| Graders deterministic | 0/6 | **FAIL**: Graders are trivially deterministic because they ignore all input. `grade_easy` at `env.py:70-71` always returns 0.8. `grade_medium` at `env.py:74-75` always returns 0.6. |
| Hard task challenging | 0/6 | No hard task exists. Cannot evaluate frontier model challenge. |

### 3. Environment Design: 11/20
| Check | Points | Evidence |
|-------|--------|----------|
| `reset()` clean | 4/5 | `reset()` at `env.py:35-53` reinitializes `self.tickets` and `self.idx`. Clean state reset. Minor: tickets are hardcoded, not parameterized by task. |
| Types well-designed | 4/5 | Pydantic models have typed fields with meaningful names. `Ticket`, `Observation`, `Action`, `Reward` are well-structured. `Action.priority` has constraints. Minor: no Field descriptions. |
| Dense reward | 1/5 | **Sparse**: Reward is binary (1.0 if correct, 0.0 if wrong) per step at `env.py:63`. No partial credit for close matches or priority considerations. |
| Episode boundaries | 2/5 | Episodes end when all tickets processed (2 tickets = 2 steps). Too short for meaningful agent learning. Max step count implicitly 2, which is unreasonably brief. |

### 4. Code Quality & Spec Compliance: 4/15
| Check | Points | Evidence |
|-------|--------|----------|
| `openenv validate` | 4/5 | `openenv.yaml` present with `name`, `description`, `observation_space`, `action_space`, `reward_range`. Tasks listed. Missing `version` is optional. Core methods implemented. |
| Docker | 1/4 | Dockerfile exists but references `app:app` which doesn't exist (no `app.py`). Build would fail on runtime. No `requirements.txt` file. |
| HF Space ready | 0/3 | No README.md with HF Space YAML header. No `openenv` tag. No `app.py` present. |
| Baseline script | 0/3 | No baseline script. No file reads `OPENAI_API_KEY`. No LLM integration. |

### 5. Creativity & Novelty: 0/10
**Novel domain**: NO — Customer support routing is a common example domain, though implementation is original.
**Interesting rewards**: NO — Simple binary correct/incorrect. No quality or efficiency dimensions.
**Engaging mechanics**: NO — Trivial 2-step episodes with obvious routing. No multi-turn reasoning, stateful complexity, or interesting constraints.

---

## Mandatory Fixes (Ordered by Score Impact)

### Fix #1: Add Third Task — Lost 7 pts from Section 2 (Task Quality)
**Issue**: Only 2 tasks implemented. Hackathon requires minimum 3 tasks.
**Location**: `env.py`, lines 78-82 and `openenv.yaml`, lines 13-17
**Fix**: 
```python
# Add to env.py after line 75:
def grade_hard(env):
    # Actual grading logic based on env state
    state = env.state()
    correct_routes = sum(1 for i, t in enumerate(env.tickets) if check_route_correct(t, env.actions[i]))
    priority_accuracy = calculate_priority_score(env.tickets, env.actions)
    return (correct_routes / len(env.tickets)) * 0.6 + priority_accuracy * 0.4

# Update TASKS dict:
TASKS = {
    "easy": {"grader": grade_easy, "description": "Route tickets correctly"},
    "medium": {"grader": grade_medium, "description": "Route with priority"},
    "hard": {"grader": grade_hard, "description": "Route ambiguous tickets with SLA constraints"},
}
```

```yaml
# Update openenv.yaml tasks section:
tasks:
  - id: easy
    grader: env.grade_easy
  - id: medium
    grader: env.grade_medium
  - id: hard
    grader: env.grade_hard
```

### Fix #2: Implement Meaningful Grader Logic — Lost 6 pts from Section 2 (Graders deterministic/meaningful)
**Issue**: Graders return constant values regardless of agent performance. This triggers DQ-3.
**Location**: `env.py`, lines 70-75
**Fix**: 
```python
def grade_easy(env):
    """Grade based on routing accuracy."""
    if not hasattr(env, 'history') or not env.history:
        return 0.0
    correct = sum(1 for ticket, action in env.history 
                  if ticket.category == action.route_to)
    return max(0.0, min(1.0, correct / len(env.history)))

def grade_medium(env):
    """Grade based on routing accuracy and priority assignment."""
    if not hasattr(env, 'history') or not env.history:
        return 0.0
    routing_score = grade_easy(env)
    priority_score = calculate_priority_accuracy(env.history)
    return max(0.0, min(1.0, routing_score * 0.7 + priority_score * 0.3))
```

### Fix #3: Create Baseline Inference Script — Lost 3 pts from Section 4 + DQ-4 trigger
**Issue**: No baseline script exists. Required for reproducibility validation.
**Location**: Missing file
**Fix**: 
```python
# baseline.py
import os
from openai import OpenAI
from env import TicketRouterEnv, Action, TASKS

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def run_baseline():
    results = {}
    for task_id, task_config in TASKS.items():
        env = TicketRouterEnv(task_id=task_id)
        obs = env.reset()
        done = False
        while not done:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Route this ticket: {obs.current_ticket.model_dump()}"}]
            )
            # Parse response into Action
            action = Action(route_to="billing", priority=1)  # Parse from response
            obs, reward, done, info = env.step(action)
        results[task_id] = task_config["grader"](env)
    return results

if __name__ == "__main__":
    print(run_baseline())
```

### Fix #4: Create app.py with API Endpoints — Lost 3 pts from Section 4 (HF Space)
**Issue**: No API endpoints exposed. Dockerfile references non-existent `app:app`.
**Location**: Missing file
**Fix**: 
```python
# app.py
from fastapi import FastAPI
from env import TicketRouterEnv, Action, TASKS

app = FastAPI()

@app.get("/tasks")
def get_tasks():
    return {"tasks": list(TASKS.keys())}

@app.post("/grader")
def grade(task_id: str):
    env = TicketRouterEnv(task_id=task_id)
    # Run episode...
    return {"score": TASKS[task_id]["grader"](env)}

@app.post("/baseline")
def run_baseline():
    # Trigger baseline script
    return {"status": "running"}
```

### Fix #5: Add Dense Reward Signal — Lost 4 pts from Section 3 (Environment Design)
**Issue**: Reward is binary 0/1 per step. No partial credit.
**Location**: `env.py`, line 63
**Fix**: 
```python
def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
    ticket = self.tickets[self.idx]
    correct_route = ticket.category == action.route_to
    
    # Dense reward with partial credit
    route_score = 1.0 if correct_route else 0.2 if action.route_to != "escalate" else 0.4
    priority_score = self._calculate_priority_score(ticket, action.priority)
    combined_score = route_score * 0.7 + priority_score * 0.3
    
    self.idx += 1
    done = self.idx >= len(self.tickets)
    obs = Observation(
        current_ticket=self.tickets[min(self.idx, len(self.tickets) - 1)],
        queue_length=len(self.tickets) - self.idx,
    )
    return obs, Reward(score=combined_score), done, {}
```

### Fix #6: Extend Episode Length — Lost 3 pts from Section 3 (Episode boundaries)
**Issue**: Only 2 tickets per episode. Too short for meaningful learning.
**Location**: `env.py`, lines 36-52
**Fix**: 
```python
def reset(self) -> Observation:
    # Generate more tickets based on task difficulty
    num_tickets = {"easy": 10, "medium": 20, "hard": 50}.get(self.task_id, 10)
    self.tickets = self._generate_tickets(num_tickets, self.task_id)
    self.idx = 0
    self.history = []
    return Observation(current_ticket=self.tickets[0], queue_length=len(self.tickets))
```

### Fix #7: Add README with HF Space Config — Lost 3 pts from Section 4
**Issue**: No README.md with Hugging Face Space YAML header.
**Location**: Missing file
**Fix**: 
```markdown
---
title: Ticket Router OpenEnv
emoji: "tickets"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Customer Support Ticket Router

An OpenEnv environment for training agents to route customer support tickets.

## Tasks
- **easy**: Route clearly categorized tickets
- **medium**: Route with priority assignment
- **hard**: Route ambiguous tickets under SLA constraints
```

---

## Summary

**Total Score**: 37/100
**Grade**: F
**Verdict**: This submission is **not ready** for the hackathon. The critical failure is having only 2 tasks instead of the required 3 (V-5 validation failure). Additionally, the graders return constant values regardless of agent performance (DQ-3 trigger), and there is no baseline inference script (DQ-4 trigger). The domain choice (customer support ticket routing) is valid and the core environment structure follows the OpenEnv spec, which is a strength. However, the implementation is too shallow: trivial tickets, binary rewards, and missing infrastructure (no API endpoints, no baseline, no README). To be competition-ready, add a third "hard" task, implement meaningful graders that evaluate actual agent performance, create a baseline script with OpenAI integration, and add the required API endpoints and HF Space configuration.
