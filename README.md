---
title: Print Farm Scheduler
emoji: 🏭
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
app_port: 7860
base_path: /web
pinned: false
tags:
  - openenv
---

# 🏭 Print Farm Scheduler — OpenEnv RL Environment

Enterprise 3D print farm logistics scheduler built on the **OpenEnv** framework.
An AI agent must optimally assign print jobs to a fleet of machines under constraints
including material compatibility, filament capacity, deadlines, mid-episode job arrivals,
and wear-based machine reliability.

## Architecture

This environment follows the standard **OpenEnv 3-component architecture**:

```
print_farm_scheduler/
├── models.py                           # Action + Observation (OpenEnv types)
├── client.py                           # EnvClient subclass (WebSocket)
├── __init__.py                        # Package exports
├── environment.py                     # Standalone env for inference script
├── tasks.py                           # Task definitions + graders
├── inference.py                       # Hackathon baseline runner
├── openenv.yaml                       # OpenEnv spec config
├── pyproject.toml                     # Package config
├── Dockerfile                         # Multi-stage build
├── requirements.txt                   # Dependencies
└── server/
    ├── app.py                         # FastAPI via create_app()
    ├── print_farm_environment.py      # Environment interface impl
    └── __init__.py
```

### Components

| Component | File | Base Class |
|-----------|------|------------|
| **Models** | `models.py` | `openenv.core.env_server.types.Action`, `Observation` |
| **Server** | `server/app.py` | `openenv.core.env_server.http_server.create_app()` |
| **Environment** | `server/print_farm_environment.py` | `openenv.core.env_server.interfaces.Environment` |
| **Client** | `client.py` | `openenv.core.EnvClient` |

## Quick Start

### Using Docker (recommended)

```bash
docker build -t print-farm-scheduler .
docker run -p 7860:7860 print-farm-scheduler
```

### Using `uv` (local development)

```bash
uv sync
uv run --project . server
```

The deployed Space also exposes:

- `/web` for the built-in OpenEnv web interface
- `/docs` for the OpenAPI surface
- `/health` for container health checks

### Using the Client

```python
from print_farm_scheduler import PrintFarmEnv, PrintFarmAction

env = PrintFarmEnv(base_url="http://localhost:7860")
result = env.reset(difficulty="easy", seed=42)
print(f"Machines: {len(result.observation.machines)}")
print(f"Queue: {len(result.observation.queue)}")

# Assign first job to machine 0
result = env.step(PrintFarmAction(type="assign", machine_id=0))
print(f"Reward: {result.reward}, Done: {result.done}")
print(result.observation.reward_info)
env.close()
```

### From Docker Image

```python
client = PrintFarmEnv.from_docker_image("print-farm-scheduler:latest")
try:
    result = client.reset(difficulty="medium", seed=42)
    result = client.step(PrintFarmAction(type="skip"))
finally:
    client.close()
```

## Environment Design

### Key Mechanics

- **Material families**: PLA↔PETG (low-temp) and Nylon↔TPU (specialty) are partially
  compatible. Same-family assignments incur a +1 step penalty and spool change delay.
  Different families are incompatible.
- **Spool change cooldown**: Switching materials costs 1 step of machine downtime.
- **Machine differentiation**: Each machine has unique `speed_modifier` and `max_weight_g`.
- **Wear degradation**: Reliability drops with cumulative `hours_used`. Base 95%, −2%/hr.
- **Mid-episode arrivals**: New jobs appear during the episode, visible ~2 steps ahead.
- **Terminal reward**: Episode-end bonus based on overall on-time completion rate.
- **Shared simulator core**: Both `environment.py` and `server/print_farm_environment.py`
  now use the same simulation engine, so inference and server behavior stay aligned.

### Curriculum Controls

`reset()` now accepts optional curriculum-style overrides in addition to `difficulty`,
`seed`, and `max_steps`. Useful knobs include:

- `n_machines`, `n_jobs`, `n_arrivals`, `n_materials`
- `deadline_tightness` to make deadlines looser or harsher
- `filament_scale` to simulate scarcity or abundance
- `speed_scale` and `weight_cap_scale` to widen machine diversity
- `arrival_shift`, `visible_ahead_steps`, `degradation_scale`, `min_reliability`

These overrides are surfaced back in `observation.metadata["curriculum"]`.

### Observation Metadata

Each observation now includes richer `metadata` with:

- current and average utilization
- on-time completion rate
- urgent and likely-late queue jobs
- material pressure across the queue
- score estimates for `easy`, `medium`, and `hard`

Server observations also include `reward_info` and `rubric_score` for training/debugging.

### Actions

| Action | Parameters | Description |
|--------|-----------|-------------|
| `assign` | `machine_id` | Assign first queued job to an idle machine |
| `preempt` | `machine_id` | Cancel running job, return to queue front |
| `prioritize` | `job_id` | Move a specific job to queue front |
| `skip` | — | Do nothing this step |

### Tasks

| Task | Difficulty | Machines | Jobs | Materials | Key Challenge |
|------|-----------|----------|------|-----------|---------------|
| Easy | ⭐ | 3 | 7 | 1 (PLA only) | Basic assignment |
| Medium | ⭐⭐ | 4 | 8+3 arrivals | 3 | Material switching + rolling arrivals |
| Hard | ⭐⭐⭐ | 6 | 14+8 arrivals | 5 | All constraints active, tight deadlines |

### Reward Components

| Component | Value | Trigger |
|-----------|-------|---------|
| Completion bonus | +0.20/job | Job finishes |
| Deadline miss | −0.10/miss | Job finishes late |
| Spool match bonus | +0.02 | Exact material match |
| Utilization bonus | up to +0.05 | Machines busy |
| Terminal bonus | up to +0.30 | Episode end (on-time rate) |
| Idle penalty | −0.01/machine | Machine sits idle |
| Preempt penalty | −0.03 | Preemption action |
| Skip penalty | −0.005 | Skip action |
| Failure penalty | −0.02/failure | Wear-based machine failure |

### Benchmark Gap

A greedy heuristic achieves **~60% on easy, ~40% on medium, ~25% on hard**.
An optimal RL policy should reach **>90% across all difficulties**.
The gap is driven by:
- **Lookahead**: heuristic is myopic; optimal policy anticipates arrivals
- **Material planning**: heuristic doesn't optimize spool change sequences
- **Wear management**: heuristic ignores cumulative machine degradation

## API Endpoints

The server automatically provides (via OpenEnv `create_app()`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment |
| `/step` | POST | Execute action |
| `/state` | GET | Current state |
| `/schema` | GET | Action/Observation JSON schemas |
| `/health` | GET | Server health check |
| `/ws` | WS | WebSocket for persistent sessions |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `meta-llama/Llama-3.1-8B-Instruct` | Model identifier |
| `HF_TOKEN` | **Yes** | — | HuggingFace API token |

## Inference

```bash
HF_TOKEN=<your-token> python inference.py
```

Runs all 3 tasks (easy/medium/hard), emitting structured `[START]`/`[STEP]`/`[END]` logs
to stdout. Falls back to heuristic if LLM fails.

## License

MIT
