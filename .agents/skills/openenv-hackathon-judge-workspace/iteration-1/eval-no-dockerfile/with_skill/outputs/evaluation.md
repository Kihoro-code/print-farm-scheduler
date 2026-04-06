# OpenEnv Official Evaluation: 35/100

## Pre-Submission Validation

| Check | Status | Notes |
|-------|--------|-------|
| V-1: HF Space deploys | ❌ | No README.md with HF Space YAML header. No `app.py` or `main.py` with Gradio/FastAPI found. |
| V-2: OpenEnv spec compliance | ✅ | `openenv.yaml` present with required fields. `step()`, `reset()`, `state()` methods implemented in `env.py` with correct signatures. |
| V-3: Dockerfile builds | ❌ | **MISSING DOCKERFILE** - No Dockerfile exists in the project directory. |
| V-4: Baseline reproduces | ❌ | No baseline inference script found. No file reads `OPENAI_API_KEY` or calls OpenAI-compatible API. |
| V-5: 3+ tasks with graders | ✅ | 3 tasks defined: `easy`, `medium`, `hard`. Each has a grader function returning values in [0.0, 1.0]. |

### Required Endpoints
| Endpoint | Status | Notes |
|----------|--------|-------|
| `/baseline` | ❌ | No route definition found. No `app.py` or `main.py` exists. |
| `/grader` | ❌ | No route definition found. No `app.py` or `main.py` exists. |
| `/tasks` | ❌ | No route definition found. No `app.py` or `main.py` exists. |

### Disqualification Gate
- DQ-1 (deploys): 🚫 — **FAIL: No Dockerfile exists.** Environment cannot be deployed without a Dockerfile.
- DQ-2 (original): ✅ — Code appears original. Data cleaning domain is not a copy of CartPole, FrozenLake, MountainCar, Taxi, or LunarLander.
- DQ-3 (graders vary): ✅ — Graders use conditional logic based on action operation and row properties. Not constant returns.
- DQ-4 (baseline script): 🚫 — **FAIL: No baseline inference script found.** No file reads API key and calls LLM.

> **Disqualified: YES** — V-3 (Dockerfile) and V-4 (Baseline script) failed. DQ-1 triggered due to missing Dockerfile.

---

## Scoring Breakdown

### 1. Real-World Utility: 18/30 — Band C
**Domain**: Data Cleaning / ETL
**Assessment**: Data cleaning is a valid real-world domain that human knowledge workers perform regularly. The operations (keep, drop, impute_mean, impute_median, cap_outliers) map to real decisions a data engineer would make. However, the implementation is shallow — the data is synthetic with simple null patterns, and the scoring logic is oversimplified. Would benefit from realistic datasets and more nuanced quality metrics.

### 2. Task & Grader Quality: 16/25
| Check | Points | Evidence |
|-------|--------|----------|
| 3+ tasks | 7/7 | 3 tasks found: `easy`, `medium`, `hard` defined in `TASKS` dict at `env.py:87-91` |
| Grader range [0.0–1.0] | 6/6 | Graders return averaged scores from step rewards which are constrained by `Reward.score` field with `ge=0.0, le=1.0` at `env.py:26` |
| Graders deterministic | 6/6 | No `random` calls in graders. `grade_easy`, `grade_medium`, `grade_hard` are pure functions of env state. |
| Hard task challenging | -3/6 | Hard task just multiplies by 0.7 (`env.py:83-84`). This scales difficulty artificially rather than presenting a genuinely harder problem. A frontier model would trivially achieve high scores. |

### 3. Environment Design: 14/20
| Check | Points | Evidence |
|-------|--------|----------|
| `reset()` clean | 5/5 | `reset()` at `env.py:37-44` reinitializes `self.data`, `self.idx`, and `self.cleaned`. No state leakage. |
| Types well-designed | 4/5 | Good Pydantic models with typed fields (`DataRow`, `Observation`, `Action`, `Reward`). Field descriptions via `Field()` on some. `Action.operation` uses `str` instead of `Literal` enum which is slightly vague. |
| Dense reward | 3/5 | Partial reward per step (0.8 or 0.3 based on action appropriateness at `env.py:49-53`). However, only two reward values — more granularity would help. |
| Episode boundaries | 2/5 | Episode ends when all 10 rows processed. Fixed at 10 rows regardless of task. Max steps reasonable but inflexible. |

### 4. Code Quality & Spec Compliance: 4/15
| Check | Points | Evidence |
|-------|--------|----------|
| `openenv validate` | 4/5 | `openenv.yaml` present with `name`, `description`, `observation_space`, `action_space`, `reward_range`. Missing `version` field in spec (has it as 1.0.0 but may not be required). Core methods implemented correctly. |
| Docker | 0/4 | **MISSING: No Dockerfile exists.** Cannot build or run via Docker. |
| HF Space ready | 0/3 | **MISSING: No README.md with HF Space YAML header.** No `app.py` or `main.py`. Not tagged with `openenv`. |
| Baseline script | 0/3 | **MISSING: No baseline script.** No file reads `OPENAI_API_KEY` from env or calls OpenAI-compatible API. |

### 5. Creativity & Novelty: 5/10
**Novel domain**: YES — Data cleaning/ETL for ML pipelines is not well-represented in existing OpenEnv or Gym environments. This is a genuine gap.
**Interesting rewards**: PARTIAL — Rewards based on operation appropriateness (impute vs keep for null rows) is reasonable but shallow. Doesn't capture data quality metrics like distribution preservation.
**Engaging mechanics**: NO — Simple row-by-row processing without multi-turn reasoning or complex state dependencies. No interesting constraints or trade-offs.

---

## Mandatory Fixes (Ordered by Score Impact)

### Fix #1: Missing Dockerfile — Lost 4 pts from Section 4, Triggers DQ-1
**Issue**: No Dockerfile exists, making the environment impossible to deploy or run in containers.
**Location**: Missing file `Dockerfile`
**Fix**: 
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
```

### Fix #2: Missing Baseline Script — Lost 3 pts from Section 4, Triggers DQ-4
**Issue**: No baseline inference script exists that reads OPENAI_API_KEY and calls an LLM to produce baseline scores.
**Location**: Missing file `baseline.py`
**Fix**: 
```python
import os
from openai import OpenAI
from env import DataCleaningEnv, Action, TASKS

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def run_baseline(task_id: str) -> float:
    env = DataCleaningEnv(task_id=task_id)
    obs = env.reset()
    done = False
    
    while not done:
        prompt = f"Data row: {obs.current_row.model_dump()}. Choose operation: keep, drop, impute_mean, impute_median, cap_outliers"
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        operation = response.choices[0].message.content.strip().lower()
        if operation not in ["keep", "drop", "impute_mean", "impute_median", "cap_outliers"]:
            operation = "keep"
        action = Action(operation=operation)
        obs, reward, done, _ = env.step(action)
    
    return TASKS[task_id]["grader"](env)

if __name__ == "__main__":
    for task_id in TASKS:
        score = run_baseline(task_id)
        print(f"{task_id}: {score:.3f}")
```

### Fix #3: Missing HF Space Configuration — Lost 3 pts from Section 4
**Issue**: No README.md with HF Space YAML header and no app.py for deployment.
**Location**: Missing files `README.md` and `app.py`
**Fix**: 
```markdown
---
title: Data Cleaning Environment
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
tags:
  - openenv
---

# Data Cleaning Environment

An OpenEnv environment for training agents on data cleaning tasks.
```

```python
# app.py
import gradio as gr
from fastapi import FastAPI
from env import DataCleaningEnv, Action, TASKS

app = FastAPI()

@app.get("/tasks")
def get_tasks():
    return {"tasks": list(TASKS.keys())}

@app.post("/grader")
def grade(task_id: str, env_state: dict):
    env = DataCleaningEnv(task_id=task_id)
    # Restore state and grade
    return {"score": TASKS[task_id]["grader"](env)}

# Gradio interface for HF Spaces
demo = gr.Interface(fn=lambda: "Data Cleaning Env", inputs=[], outputs="text")
demo.launch()
```

### Fix #4: Missing requirements.txt — Lost implicit points (Docker can't build)
**Issue**: No requirements.txt to specify dependencies.
**Location**: Missing file `requirements.txt`
**Fix**: 
```text
pydantic>=2.0.0
openai>=1.0.0
gradio>=4.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
```

### Fix #5: Hard Task Not Challenging — Lost 3 pts from Section 2
**Issue**: Hard task just multiplies score by 0.7 rather than presenting a genuinely harder problem.
**Location**: `env.py:83-84`
**Fix**: 
```python
def grade_hard(env):
    """Hard task: penalize incorrect operations more heavily and require optimal choices"""
    base_score = sum(c[2] for c in env.cleaned) / max(1, len(env.cleaned))
    # Penalize any non-optimal operations
    optimal_ops = sum(1 for c in env.cleaned if c[0].has_nulls and c[1].operation in ["impute_mean", "impute_median"])
    optimal_ratio = optimal_ops / max(1, sum(1 for c in env.cleaned if c[0].has_nulls))
    return base_score * optimal_ratio * 0.9  # Must be nearly optimal to score well
```

### Fix #6: Action.operation Should Use Literal Type — Lost 1 pt from Section 3
**Issue**: `Action.operation` uses `str` which allows any value. Should be constrained.
**Location**: `env.py:21`
**Fix**: 
```python
from typing import Literal

class Action(BaseModel):
    operation: Literal["keep", "drop", "impute_mean", "impute_median", "cap_outliers"]
    column_indices: List[int] = Field(default_factory=list)
```

---

## Summary

**Total Score**: 35/100
**Grade**: F
**Verdict**: This submission is **not ready for competition** and is **disqualified** due to missing critical infrastructure. The core environment logic is reasonable — data cleaning is a valid real-world domain, the Pydantic models are well-typed, and the graders produce variable scores. However, the submission is missing a Dockerfile (V-3 failure, triggers DQ-1), baseline inference script (V-4 failure, triggers DQ-4), HF Space configuration, and requirements.txt. Without these files, the environment cannot be deployed, validated, or benchmarked. The biggest strength is the domain choice and clean environment implementation. The critical weakness is the complete absence of deployment infrastructure — this needs a Dockerfile, app.py, baseline.py, README.md, and requirements.txt before it can be evaluated in the hackathon pipeline.
