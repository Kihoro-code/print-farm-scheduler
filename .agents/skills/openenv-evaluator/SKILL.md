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

## Agent Initialization Protocol
Before scoring anything, complete these steps in order using your shell tool:

1. Run `find . -type f` to list all files in the project.
2. Run `cat` on every `.py` file found.
3. Run `cat openenv.yaml Dockerfile README.md requirements.txt 2>/dev/null` to read supporting files.
4. Identify the primary environment file — the one defining Pydantic models: Observation, Action, Reward.
5. Identify all task files and grader implementations.
6. Identify `inference.py` in the root directory — this is the required inference script name.

If no Python files exist: output "❌ EVALUATION ABORTED: No .py files found." and stop.
Score only what is **implemented in code**. Do not reward stated intent in README.

---

## Phase 1: Pre-Submission Checklist Gate (Run First)
All five must pass. Any failure → output "🚫 DISQUALIFIED" with reason and stop.

| # | Check | How to Verify |
|---|-------|---------------|
| DQ-1 | Environment deploys and responds | Dockerfile exists; `openenv.yaml` present with correct schema; HF Space config in README YAML header |
| DQ-2 | OpenEnv spec compliance | `openenv.yaml` present; typed Observation/Action/Reward Pydantic models; `step()`/`reset()`/`state()` implemented |
| DQ-3 | Graders never constant | Read each grader — flag if output is hardcoded or has no branching logic |
| DQ-4 | `inference.py` exists and is correct | File named exactly `inference.py` in root; reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from env; uses OpenAI client; emits `[START]`/`[STEP]`/`[END]` structured stdout logs; completes in <20min on vcpu=2/8gb RAM |
| DQ-5 | Not plagiarized | Compare concept against CartPole, FrozenLake, MountainCar, Taxi, LunarLander — flag if <20% novel |

Mark each as ✅ PASS or 🚫 FAIL with filename and line number.

---

## inference.py Compliance Detail (for DQ-4)
The inference script has strict requirements — any deviation causes automated scoring failure:

**Required environment variables (read via `os.environ`):**
- `API_BASE_URL` — the LLM API endpoint
- `MODEL_NAME` — the model identifier
- `HF_TOKEN` — Hugging Face / API key

**Required stdout log format — exact field names and ordering:**
```
[START] {"task": "<task_name>", "episode": <n>}
[STEP] {"step": <n>, "action": <...>, "reward": <float>, "done": <bool>}
[END] {"task": "<task_name>", "score": <float>}
```
Any deviation in field names, ordering, or format = automated scoring failure.

**Infrastructure constraints:**
- Runtime must complete in under 20 minutes
- Must run on vcpu=2, memory=8gb — do not use heavy models or large batches

**OpenAI client usage:**
```python
from openai import OpenAI
client = OpenAI(
    api_key=os.environ["HF_TOKEN"],
    base_url=os.environ["API_BASE_URL"]
)
model = os.environ["MODEL_NAME"]
```

---

## Scoring Rubric (100 pts total)

### 1. Real-World Utility — 30 pts
The environment must simulate a task **humans actually do**. Not games, not toys.
Valid domains: email triage, code review, data cleaning, scheduling, customer support, content moderation.

Score exactly one band:

| Band | Score | Criteria |
|------|-------|----------|
| F | 0–5 | Toy or artificial problem. No practical application. A game, simulation, or abstract puzzle. |
| D | 6–15 | Valid domain but shallow modeling. Real label, fake mechanics (e.g., "email triage" that's just random label assignment). |
| C | 16–25 | Good domain modeling with realistic inputs, actions, and outcomes. Would be useful for evaluating agents on this task. |
| A | 26–30 | Excellent. Fills a real gap in the RL/agent community. Immediate value — someone would use this to benchmark a production agent today. |

### 2. Task & Grader Quality — 25 pts
Must have **minimum 3 tasks** with agent graders. Graders must score 0.0–1.0. Tasks must span easy → medium → hard.

| Sub-check | Points | Verification |
|-----------|--------|-------------|
| At least 3 tasks implemented | 7 | Count task definitions in code |
| Graders produce scores in [0.0, 1.0] | 6 | Check return values and clipping logic |
| Graders are deterministic and reproducible | 6 | Same input must always produce same score — check for unseeded randomness inside graders |
| Hard task genuinely challenges frontier models | 6 | Read the hard task — would GPT-4 / Claude score <0.7 on a naive attempt? |

### 3. Environment Design — 20 pts

| Sub-check | Points | Verification |
|-----------|--------|-------------|
| `reset()` produces a clean, independent state | 5 | Check that `reset()` reinitialises all state variables and returns initial observation |
| Action/Observation types are well-designed and documented | 5 | Check Pydantic models — are fields typed, named, and meaningful? |
| Reward function provides dense, varying signal (not just binary end-of-episode) | 5 | Is partial progress rewarded? Are destructive actions like infinite loops penalized? |
| Episode boundaries are sensible | 5 | Do episodes end at a logical point? Is max step count reasonable? |

### 4. Code Quality & Spec Compliance — 15 pts
Pass/fail per item — points only if fully working.

| Sub-check | Points | Verification |
|-----------|--------|-------------|
| `openenv validate` passes | 5 | openenv.yaml present; typed Observation/Action/Reward models; `step()`/`reset()`/`state()` all implemented correctly |
| `docker build && docker run` works | 4 | Dockerfile present, valid base image, all pip installs present, entry point defined |
| HF Space deployment ready | 3 | HF Space config in README YAML header, tagged with `openenv`, containerized |
| `inference.py` runs and reproduces scores | 3 | Correct env vars, correct log format, runs on vcpu=2/8gb in <20min |

### 5. Creativity & Novelty — 10 pts

| Sub-check | Points | Verification |
|-----------|--------|-------------|
| Domain not seen in OpenEnv before | 4 | Judge against known OpenEnv/OpenAI Gym/DeepMind environments |
| Reward design has interesting or non-obvious properties | 3 | Does it reward something subtle or multi-dimensional? |
| Clever mechanics that make the environment genuinely engaging | 3 | Is there a mechanic beyond a simple input→output scorer? |

---

## Required Output Format

### OpenEnv Official Evaluation: [Total]/100

**Pre-Submission Gate:**
- DQ-1 (deploys): [✅/🚫] — [reason if fail]
- DQ-2 (spec compliance): [✅/🚫] — [reason if fail]
- DQ-3 (graders vary): [✅/🚫] — [reason if fail]
- DQ-4 (inference.py): [✅/🚫] — [reason if fail, cite exact line]
- DQ-5 (original): [✅/🚫] — [reason if fail]

> Disqualified: [YES/NO]

---

**1. Real-World Utility: [Score]/30 — Band [A/C/D/F]**
*Verdict:* [Name the exact real-world task it models. State whether an agent team would actually use this to benchmark their system. If Band D or F, name what's shallow or fake about the modeling.]

**2. Task & Grader Quality: [Score]/25**
- 3+ tasks: [YES/NO — count found]
- Grader range [0.0–1.0]: [YES/NO — cite file and line if broken]
- Graders deterministic: [YES/NO — cite any unseeded randomness]
- Hard task challenges frontier models: [YES/NO — explain why or why not]

**3. Environment Design: [Score]/20**
- `reset()` clean: [YES/NO]
- Action/Observation typing: [YES/NO — note missing or vague fields]
- Dense reward: [YES/NO — describe the reward signal quality]
- Episode boundaries sensible: [YES/NO]

**4. Code Quality & Spec Compliance: [Score]/15**
- `openenv validate`: [PASS/FAIL — list missing fields]
- Docker: [PASS/FAIL — list missing dependencies or config]
- HF Space ready: [PASS/FAIL]
- `inference.py`: [PASS/FAIL — list any env var, log format, or runtime violations]

**5. Creativity & Novelty: [Score]/10**
*Verdict:* [Is this a domain gap? What's the most interesting mechanic? What's the most boring part?]

---

### 🔨 Mandatory Fixes (Ordered by Score Impact)
For each issue that cost points:
- **Points lost:** N pts from Section X
- **Issue:** One sentence.
- **Location:** `filename.py`, line N (or "missing file" if absent)
- **Fix:** A self-contained code snippet or exact file/field that resolves it. No generic advice.