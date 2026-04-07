---
title: Print Farm Scheduler
emoji: "🖨️"
colorFrom: blue
colorTo: green
sdk: docker
app_file: inference.py
tags:
  - openenv
pinned: false
---

# Print Farm Scheduler

An OpenEnv reinforcement learning environment simulating an enterprise 3D printing facility. The agent must optimally assign print jobs to a fleet of heterogeneous machines under material compatibility families, filament capacity, deadline pressure, spool-change delays, machine wear degradation, and build-weight constraints.

## Motivation

Scheduling print jobs across a multi-machine farm is a real NP-hard constrained optimization problem faced daily by contract manufacturers and rapid-prototyping labs. This environment captures the key difficulties:

- **Material families** — PLA↔PETG and Nylon↔TPU are partially compatible (printable with a quality penalty); ABS requires exact match
- **Spool change delay** — switching materials costs 1 step of downtime before printing begins
- **Filament capacity** — machines have finite remaining filament that depletes with each assignment
- **Deadline pressure** — jobs have customer deadlines; missing them is costly
- **Machine differentiation** — machines vary in print speed (`speed_modifier`) and max build weight (`max_weight_g`)
- **Wear degradation** — machine reliability starts at 95% and drops ~2% per hour of printing (minimum 50%)
- **Preemption tradeoffs** — canceling a running job wastes progress but may free a machine for something urgent
- **Partial observability** — future job arrivals are hidden until 2 steps before they appear, simulating advance customer bookings that are confirmed shortly before production cutoff

## Action Space

| Action | Fields | Description |
|--------|--------|-------------|
| `assign` | `machine_id` | Assign the first job in queue to an idle, compatible machine |
| `preempt` | `machine_id` | Cancel the job currently printing or in spool change; returned to front of queue |
| `prioritize` | `job_id` | Move a specific job to the front of the queue |
| `skip` | — | Do nothing this step |

## Observation Space

Each observation contains:
- `machines[]` — status (`idle`/`printing`/`changing_spool`), loaded material, filament remaining, speed modifier, max build weight, cumulative hours used, current job progress
- `queue[]` — pending jobs with material, weight, print time, deadline
- `pending_arrivals[]` — advance customer bookings visible within the next 2 steps (simulates confirmed orders not yet in production queue)
- `completed_count`, `deadlines_missed`, `total_jobs_ever`

## Material Compatibility

| Loaded → Required | PLA | PETG | ABS | Nylon | TPU |
|-------------------|-----|------|-----|-------|-----|
| **PLA** | ✅ Exact | ⚠️ Partial | ❌ | ❌ | ❌ |
| **PETG** | ⚠️ Partial | ✅ Exact | ❌ | ❌ | ❌ |
| **ABS** | ❌ | ❌ | ✅ Exact | ❌ | ❌ |
| **Nylon** | ❌ | ❌ | ❌ | ✅ Exact | ⚠️ Partial |
| **TPU** | ❌ | ❌ | ❌ | ⚠️ Partial | ✅ Exact |

- ✅ **Exact match**: prints immediately, earns spool-match bonus
- ⚠️ **Partial match**: +1 extra print step + 1-step spool change delay
- ❌ **Incompatible**: cannot assign

## Tasks

| Task | Machines | Materials | Jobs | Arrivals | Speed Range | Wear | Max Steps |
|------|----------|-----------|------|----------|-------------|------|-----------|
| **Easy** | 3 | 1 (PLA) | 7 | 0 | 0.95–1.05× | Yes | 30 |
| **Medium** | 4 | 3 (PLA, PETG, ABS) | 8 | 3 at step ~10 | 0.8–1.2× | Yes | 30 |
| **Hard** | 6 | 5 (PLA, PETG, ABS, Nylon, TPU) | 14 | 8 in waves | 0.6–1.4× | Yes | 30 |

### Grading

- **Easy**: completion rate (jobs on-time / total jobs)
- **Medium**: `0.7 × completion_rate + 0.3 × utilization`
- **Hard**: `0.7 × completion_rate + 0.2 × utilization + 0.1 × preemption_efficiency`

## Setup & Usage

```bash
# Build
docker build -t print-farm-scheduler .

# Run with required environment variables
docker run -e HF_TOKEN=your_token \
           -e API_BASE_URL=https://api-inference.huggingface.co/v1 \
           -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
           print-farm-scheduler
```

Or run locally:

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
python inference.py
```

## Reward Design

Rewards are normalized to [-1, 0, 1] with named breakdown components:

| Component | Value | Trigger |
|-----------|-------|---------|
| `completion_bonus` | +0.20 × n | Job completed (on-time or late) |
| `utilization_bonus` | +0.05 × ratio | Active machines / total machines |
| `spool_match_bonus` | +0.02 | Assign with exact material match |
| `terminal_bonus` | +0.30 × rate | Episode ends — on-time completions / total jobs |
| `idle_penalty` | -0.01 × n | Per idle machine per step |
| `preempt_penalty` | -0.03 | Preemption action |
| `deadline_miss_penalty` | -0.10 × n | Job completed after deadline |
| `partial_match_penalty` | -0.015 | Assign with partial material family match |
| `failure_penalty` | -0.02 × n | Machine failure event |
| `skip_penalty` | -0.005 | Skip action |
| `step_penalty` | -0.001 | Every step (anti-loop) |

## Machine Differentiation

Each machine is generated with unique attributes:

| Attribute | Easy Range | Medium Range | Hard Range |
|-----------|-----------|-------------|-----------|
| `speed_modifier` | 0.95–1.05 | 0.8–1.2 | 0.6–1.4 |
| `max_weight_g` | 90–100g | 65–100g | 50–100g |

- **`speed_modifier`**: scales effective print steps — a 1.2× machine completes a 3-step job in ~2 steps
- **`max_weight_g`**: limits which jobs can be assigned — a 60g machine cannot print a 75g job

## Wear-Based Degradation

Machine reliability is no longer a flat constant. It degrades with cumulative printing hours:

```
reliability = max(0.50, 0.95 - 0.02 × hours_used)
```

- Fresh machines start at 95% reliability per step
- After 10 hours of printing: 75% reliability
- After 20 hours: 55% reliability
- Minimum reliability floor: 50%

This creates a load-balancing incentive — agents must spread work across machines to avoid wearing down their best printers.

## Benchmark Gap Filled

No existing RL benchmark (OpenAI Gym, DeepMind Control Suite, AgentBench, WebArena, OR-Gym) models multi-machine manufacturing scheduling with all of:
- Heterogeneous machine capabilities (speed, capacity, material compatibility)
- Stochastic equipment degradation requiring load-balancing
- Partial observability of demand (advance bookings visible ≤2 steps ahead)
- Material family compatibility creating non-trivial assignment constraints

This fills the gap between abstract job-shop scheduling (OR-Gym) and real manufacturing operations research. Any team building LLM-based scheduling agents for factories can immediately use this as a standard evaluation benchmark.

## Why This Challenges Frontier Models

The hard task creates a search space of ~6^22 possible assignment sequences (22 jobs × 6 machines), compounded by:
- **Material constraints**: only 2 of 6 machines may be compatible with any given job
- **Temporal coupling**: assigning job A now may prevent job B later (filament depletion)
- **Stochastic failures**: wear-based degradation means plans become invalid mid-episode
- **Multi-wave arrivals**: 8 jobs arrive mid-episode, invalidating earlier scheduling decisions
- **Filament scarcity**: 40–120g per machine vs 10–60g per job = only ~2–3 jobs per machine before refill impossible

A greedy heuristic achieves ~0.35 on the hard task. An optimal scheduler would require lookahead planning across all 30 steps with stochastic uncertainty — well beyond naive LLM prompt-and-respond loops.

## Files

| File | Description |
|------|-------------|
| `environment.py` | Core environment: Pydantic models, state machine, reward logic |
| `tasks.py` | Task definitions and deterministic graders |
| `inference.py` | LLM baseline runner with [START]/[STEP]/[END] logging |
| `openenv.yaml` | OpenEnv spec manifest |
| `Dockerfile` | Container build |
| `requirements.txt` | Python dependencies |

