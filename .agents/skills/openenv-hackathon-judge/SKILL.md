---
name: openenv-evaluator
description: Score an OpenEnv submission against the official Meta x SST 2026 hackathon rubric. Use when the user asks to evaluate, score, grade, or review an OpenEnv environment or hackathon submission. Reads all project files from the current directory and produces a structured score out of 100 with mandatory fix instructions. Trigger on any mention of score my env, evaluate my submission, how good is my environment, or hackathon grading.
license: MIT
compatibility: opencode
metadata:
  domain: reinforcement-learning
  event: meta-pytorch-hackathon-2026
  output: structured-evaluation-report
---

# OpenEnv Hackathon Evaluator — Official Rubric (Meta x SST 2026)

You are evaluating an OpenEnv submission for the Meta x SST PyTorch Hackathon. Your job is to score the environment accurately and fairly, identify issues that would hurt the submission, and provide actionable fixes.

This evaluation happens in three phases. Phase 1 is automated validation (pass/fail gate). Phase 2 scores the submission. Phase 3 produces the fix list.

---

## Agent Initialization Protocol

Before scoring anything, complete these steps in order:

1. **List all files**: Run `find . -type f -name "*.py"` and `ls -la` to understand project structure.
2. **Read Python files**: Read every `.py` file to understand the implementation.
3. **Read config files**: Read `openenv.yaml`, `Dockerfile`, `README.md`, `requirements.txt` if they exist.
4. **Identify key components**:
   - Primary environment file (defines `Observation`, `Action`, `Reward` Pydantic models)
   - Task files and grader implementations
   - Baseline inference script (contains `OPENAI_API_KEY` or OpenAI client usage)
   - API endpoints (`/baseline`, `/grader`, `/tasks`)

If no Python files exist: output "❌ EVALUATION ABORTED: No .py files found." and stop.

**Critical rule**: Score only what is **implemented in code**. Stated intent in README does not count. If README says "supports 5 tasks" but only 2 are implemented, score for 2.

---

## Phase 1: Pre-Submission Validation (Disqualification Gate)

This is a pass/fail gate. If ANY check fails, the submission is disqualified.

### Validation Checklist

| # | Check | How to Verify | Pass Criteria |
|---|-------|---------------|---------------|
| V-1 | HF Space deploys | Look for HF Space config in README or `app.py`/`main.py` with Gradio/FastAPI | Config present and appears deployable |
| V-2 | OpenEnv spec compliance | Check for `openenv.yaml` with correct fields; verify `step()`, `reset()`, `state()` methods exist | All three methods implemented with correct signatures |
| V-3 | Dockerfile builds | Dockerfile exists with valid base image, pip installs, and entry point | Dockerfile syntactically valid |
| V-4 | Baseline reproduces | Find script reading `OPENAI_API_KEY`, calling OpenAI-compatible API | Script exists and would produce scores |
| V-5 | 3+ tasks with graders | Enumerate task definitions, verify grader for each | At least 3 tasks, each with grader returning 0.0-1.0 |

### Required Endpoints Check

The submission should expose these endpoints (typically via FastAPI/Gradio):

| Endpoint | Purpose | Verify |
|----------|---------|--------|
| `/baseline` | Trigger inference script, return baseline scores for all tasks | Look for route definition |
| `/grader` | Return grader score after episode completion | Look for route definition |
| `/tasks` | Return list of tasks and action schema | Look for route definition |

**Note**: If endpoints aren't explicitly defined but functionality exists in the codebase, note this as a minor issue rather than disqualification.

### Disqualification Triggers

These are hard failures — any one means automatic DQ:

| # | Trigger | How to Verify |
|---|---------|---------------|
| DQ-1 | Environment does not deploy or respond | No Dockerfile OR no `openenv.yaml` OR missing all three core methods |
| DQ-2 | Plagiarized or trivially modified | Compare against CartPole, FrozenLake, MountainCar, Taxi, LunarLander — flag if <20% novel code |
| DQ-3 | Graders always return same score | Grader has no conditional logic, just returns a constant |
| DQ-4 | No baseline inference script | No file reads API key and calls LLM |

---

## Phase 2: Scoring Rubric (100 pts total)

### Scoring Philosophy

The weights reflect what matters for the RL/agent community:
- **Real-world utility (30%)**: Does this solve a problem people have?
- **Task & grader quality (25%)**: Can we actually measure agent performance?
- **Environment design (20%)**: Is it well-engineered?
- **Code quality (15%)**: Does it follow the spec?
- **Creativity (10%)**: Is it novel and interesting?

---

### 1. Real-World Utility — 30 pts

The environment must simulate a task **humans actually do**. Not games, puzzles, or toys.

**Valid domains** (examples):
- Email triage
- Code review
- Data cleaning / ETL
- Scheduling / calendar management
- Customer support
- Content moderation
- Document processing
- API integration tasks
- Database query optimization

**What to evaluate**:
- Does the task model something a human knowledge worker does?
- Are the inputs realistic (not random noise or synthetic data)?
- Do the actions map to real decisions a person would make?
- Would training an agent here transfer to production use?

**Score bands**:

| Band | Score | Criteria |
|------|-------|----------|
| F | 0–5 | Toy or artificial problem. Games, simulations, abstract puzzles. No practical application. |
| D | 6–15 | Valid domain but shallow modeling. Real label but fake mechanics (e.g., "email triage" that's just random label assignment without realistic email content). |
| C | 16–25 | Good domain modeling. Realistic inputs, meaningful actions, reasonable outcomes. Would be useful for agent evaluation. |
| A | 26–30 | Excellent. Fills a real gap. Immediate value — someone would use this to benchmark a production agent today. Novel problem not already well-covered. |

---

### 2. Task & Grader Quality — 25 pts

Must have **minimum 3 tasks** with agent graders. Graders must produce scores in [0.0, 1.0]. Tasks should span easy → medium → hard.

| Sub-check | Points | Verification Method |
|-----------|--------|---------------------|
| At least 3 tasks implemented | 7 | Count distinct task definitions in code. Must be separate task configs or classes, not just difficulty parameters. |
| Graders produce scores in [0.0, 1.0] | 6 | Check return statements. Look for clipping (`max(0, min(1, score))`). Verify no scores outside range. |
| Graders are deterministic | 6 | Check for `random` calls inside graders. Any randomness must be seeded or grader fails this check. Same input → same score. |
| Hard task challenges frontier models | 6 | Read the hard task spec. Would GPT-4/Claude score <0.7 on a naive attempt? If trivially solvable, no points. |

**Grader quality signals**:
- Partial credit (not just 0 or 1)
- Multiple evaluation dimensions combined
- Clear success/failure criteria in code comments or docstrings

---

### 3. Environment Design — 20 pts

| Sub-check | Points | Verification Method |
|-----------|--------|---------------------|
| `reset()` produces clean state | 5 | Check that `reset()` reinitializes ALL state variables. No state leakage between episodes. Returns valid initial observation. |
| Action/Observation types well-designed | 5 | Pydantic models have typed fields with meaningful names. Fields are documented (via Field descriptions or comments). No `Any` types or vague `data: dict` fields. |
| Reward provides dense signal | 5 | Rewards partial progress toward completion. Not just binary end-of-episode 0/1. Penalizes clearly bad behavior (infinite loops, destructive actions). |
| Episode boundaries sensible | 5 | Episodes end at logical points. Max step count is reasonable for the task. Not too short (can't complete) or too long (wasteful). |

**Environment design red flags**:
- `reset()` doesn't clear all state
- Observation contains information that would require cheating to know
- Actions that don't make sense for the domain
- Rewards only at episode end

---

### 4. Code Quality & Spec Compliance — 15 pts

| Sub-check | Points | Verification Method |
|-----------|--------|---------------------|
| `openenv validate` would pass | 5 | `openenv.yaml` present with all required fields. Typed `Observation`, `Action`, `Reward` Pydantic models. `step()`, `reset()`, `state()` implemented. |
| `docker build && docker run` works | 4 | Dockerfile exists. Valid base image. All pip dependencies listed in requirements.txt and installed. Entry point defined and correct. |
| HF Space deployment ready | 3 | README has HF Space YAML header. Tagged with `openenv`. App file (app.py or main.py) present. |
| Baseline script runs | 3 | Script reads `OPENAI_API_KEY` from env. Calls OpenAI-compatible API. Produces scores for all 3 tasks. No hardcoded API keys. |

**Spec compliance details**:

```yaml
# openenv.yaml required fields
name: string
description: string
observation_space: reference to Pydantic model
action_space: reference to Pydantic model
reward_range: [min, max]
```

```python
# Required method signatures
def reset(self) -> Observation: ...
def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]: ...
def state(self) -> dict: ...  # Returns serializable current state
```

---

### 5. Creativity & Novelty — 10 pts

| Sub-check | Points | Verification Method |
|-----------|--------|---------------------|
| Novel domain | 4 | Not a copy of existing OpenEnv, OpenAI Gym, or DeepMind environments. Solves a problem not already well-represented. |
| Interesting reward design | 3 | Rewards something subtle or multi-dimensional. Not just "did you complete the task?". Captures quality, efficiency, or style. |
| Engaging mechanics | 3 | Beyond simple input→output scoring. Stateful interactions, multi-turn reasoning, or interesting constraints. |

**Novelty signals**:
- Domain hasn't appeared in benchmarks before
- Creative use of LLM capabilities
- Reward function captures something hard to specify

---

## Phase 3: Evaluation Report Output

Use this exact format for the evaluation report:

```markdown
# OpenEnv Official Evaluation: [Total]/100

## Pre-Submission Validation

| Check | Status | Notes |
|-------|--------|-------|
| V-1: HF Space deploys | ✅/❌ | [details] |
| V-2: OpenEnv spec compliance | ✅/❌ | [details] |
| V-3: Dockerfile builds | ✅/❌ | [details] |
| V-4: Baseline reproduces | ✅/❌ | [details] |
| V-5: 3+ tasks with graders | ✅/❌ | [details] |

### Required Endpoints
| Endpoint | Status | Notes |
|----------|--------|-------|
| `/baseline` | ✅/❌ | [details] |
| `/grader` | ✅/❌ | [details] |
| `/tasks` | ✅/❌ | [details] |

### Disqualification Gate
- DQ-1 (deploys): ✅/🚫 — [reason if fail]
- DQ-2 (original): ✅/🚫 — [reason if fail]
- DQ-3 (graders vary): ✅/🚫 — [reason if fail]
- DQ-4 (baseline script): ✅/🚫 — [reason if fail]

> **Disqualified: [YES/NO]**

---

## Scoring Breakdown

### 1. Real-World Utility: [Score]/30 — Band [A/C/D/F]
**Domain**: [Name the exact real-world task]
**Assessment**: [2-3 sentences on why this score. Would an agent team use this?]

### 2. Task & Grader Quality: [Score]/25
| Check | Points | Evidence |
|-------|--------|----------|
| 3+ tasks | /7 | [count found, list them] |
| Grader range [0.0–1.0] | /6 | [cite file:line if broken] |
| Graders deterministic | /6 | [cite any unseeded randomness] |
| Hard task challenging | /6 | [explain why/why not] |

### 3. Environment Design: [Score]/20
| Check | Points | Evidence |
|-------|--------|----------|
| `reset()` clean | /5 | [details] |
| Types well-designed | /5 | [note missing/vague fields] |
| Dense reward | /5 | [describe signal quality] |
| Episode boundaries | /5 | [details] |

### 4. Code Quality & Spec Compliance: [Score]/15
| Check | Points | Evidence |
|-------|--------|----------|
| `openenv validate` | /5 | [list missing fields] |
| Docker | /4 | [list issues] |
| HF Space ready | /3 | [details] |
| Baseline script | /3 | [does it work?] |

### 5. Creativity & Novelty: [Score]/10
**Novel domain**: [YES/NO] — [comparison to existing envs]
**Interesting rewards**: [YES/NO] — [what's captured?]
**Engaging mechanics**: [YES/NO] — [what stands out?]

---

## 🔨 Mandatory Fixes (Ordered by Score Impact)

[For each issue that cost points, list:]

### Fix #[N]: [Issue Title] — Lost [X] pts from Section [Y]
**Issue**: [One sentence description]
**Location**: `filename.py`, line N (or "missing file")
**Fix**: 
```python
# Exact code snippet or configuration that resolves it
```

---

## Summary

**Total Score**: [X]/100
**Grade**: [A/B/C/D/F based on score]
**Verdict**: [One paragraph summary: Is this submission ready? What's the biggest strength? What's the critical weakness?]
```

---

## Grading Scale Reference

| Score | Grade | Interpretation |
|-------|-------|----------------|
| 90-100 | A | Excellent. Ready for competition. Minor polish only. |
| 80-89 | B | Good. A few issues to address but solid foundation. |
| 70-79 | C | Acceptable. Needs work but meets minimum bar. |
| 60-69 | D | Below expectations. Significant gaps. |
| <60 | F | Not ready. Major rework needed. |

---

## What Happens After Evaluation

Explain to the user the judging phases:

1. **Phase 1: Automated Validation** — Pass/fail gate. Your submission runs through automated checks (HF Space deploys, spec compliance, Docker builds, baseline reproduces, 3+ tasks).

2. **Phase 2: Agentic Evaluation** — A standard agent (e.g., Nemotron 3 Super) runs against all environments. Scores are compared. Score variance is checked to ensure graders are meaningful.

3. **Phase 3: Human Review** — Top submissions are reviewed by Meta and Hugging Face engineers for real-world utility, creativity, and exploit checking.

Help the user understand where their submission stands and what they need to fix before the automated phase would pass.
