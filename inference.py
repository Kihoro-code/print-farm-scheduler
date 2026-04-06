"""
inference.py — OpenEnv hackathon baseline inference script.

Must be in the project root. Reads credentials from environment variables.
Emits structured stdout logs in [START]/[STEP]/[END] format.
Must complete all 3 tasks in under 20 minutes on vcpu=2, memory=8gb.
"""

import os
import json

from openai import OpenAI

from environment import PrintFarmEnv, Action
from tasks import TASKS


# ── Required env vars — do NOT rename these ──────────────────────────────────
client = OpenAI(
    api_key=os.environ["HF_TOKEN"],
    base_url=os.environ["API_BASE_URL"],
)
MODEL_NAME = os.environ["MODEL_NAME"]


# ── LLM System Prompt ────────────────────────────────────────────────────────

LLM_SYSTEM_PROMPT = """You are a 3D print farm scheduling agent. At each step you receive
the current state of the factory and must choose ONE action.

Actions (respond with ONLY a JSON object, no explanation):
- {"type": "assign", "machine_id": <int>} — assign the FIRST job in queue to an idle machine
- {"type": "preempt", "machine_id": <int>} — cancel a running job on a machine (returned to front of queue)
- {"type": "prioritize", "job_id": <int>} — move a specific job to the front of the queue
- {"type": "skip"} — do nothing this step

Rules:
- Material families: PLA↔PETG (low-temp) and Nylon↔TPU (specialty) are partially compatible
  * Exact material match: starts printing immediately, earns spool-match bonus
  * Same-family match: adds +1 print step and 1-step spool change delay
  * Different families: INCOMPATIBLE, cannot assign
- Each machine has a speed_modifier (e.g. 1.2 = 20% faster, 0.8 = 20% slower)
- Each machine has a max_weight_g — jobs heavier than this cannot be assigned
- Machine must have enough filament (filament_remaining_g) for the job
- Machine reliability degrades with use (hours_used) — spread load to avoid wear
- New jobs may arrive mid-episode — you won't see them until ~2 steps before
- Preemption wastes the progress already made on a job
- Goal: maximize jobs completed before deadline, then maximize machine utilization

Respond with ONLY a valid JSON action object. No markdown, no explanation."""


# ── Heuristic Fallback ───────────────────────────────────────────────────────


def heuristic_action(obs, state: dict, step: int) -> Action:
    """
    Greedy heuristic fallback: prioritize urgent jobs, assign queue[0] to idle
    machines (preferring exact material match and fast machines), preempt if a
    running job blocks an urgent one, else skip.
    """
    queue = state["queue"]
    machines = state["machines"]

    # 1. If queue[0] is compatible with any idle machine → assign
    #    Prefer exact material match and faster machines
    if queue:
        first_job = queue[0]
        best_machine = None
        best_score = -1.0
        for machine in machines:
            if machine["status"] != "idle":
                continue
            # Check material compatibility
            loaded = machine["material_loaded"]
            required = first_job["material"]
            if loaded == required:
                compat_score = 2.0  # Exact match preferred
            elif _same_family(loaded, required):
                compat_score = 1.0  # Partial match acceptable
            else:
                continue  # Incompatible
            if machine["filament_remaining_g"] < first_job["weight_g"]:
                continue
            if first_job["weight_g"] > machine.get("max_weight_g", 100.0):
                continue
            # Prefer faster machines (higher speed_modifier)
            speed = machine.get("speed_modifier", 1.0)
            score = compat_score + speed * 0.5
            if score > best_score:
                best_score = score
                best_machine = machine
        if best_machine is not None:
            return Action(type="assign", machine_id=best_machine["id"])

    # 2. Try preempting if there's an urgent job waiting
    if queue:
        sorted_queue = sorted(queue, key=lambda j: j["deadline_step"])
        urgent = [
            j
            for j in sorted_queue
            if (j["deadline_step"] - step) <= j["print_steps"] + 3
        ]
        if urgent:
            for machine in machines:
                if machine["status"] not in ("printing", "changing_spool"):
                    continue
                running = machine.get("current_job")
                if running is None:
                    continue
                running_urgency = running["deadline_step"] - step
                if running_urgency > urgent[0]["deadline_step"] - step + 2:
                    return Action(type="preempt", machine_id=machine["id"])

    # 3. Prioritize a compatible job to front of queue for next step
    if queue:
        idle_machines = [m for m in machines if m["status"] == "idle"]
        for job in queue[1:]:
            for m in idle_machines:
                loaded = m["material_loaded"]
                required = job["material"]
                mat_ok = loaded == required or _same_family(loaded, required)
                weight_ok = m["filament_remaining_g"] >= job["weight_g"]
                cap_ok = job["weight_g"] <= m.get("max_weight_g", 100.0)
                if mat_ok and weight_ok and cap_ok:
                    return Action(type="prioritize", job_id=job["id"])

    # 4. Nothing useful to do
    return Action(type="skip")


def _same_family(mat_a: str, mat_b: str) -> bool:
    """Check if two materials belong to the same compatibility family."""
    families = {
        "PLA": "low_temp",
        "PETG": "low_temp",
        "ABS": "mid_temp",
        "Nylon": "specialty",
        "TPU": "specialty",
    }
    return families.get(mat_a) == families.get(mat_b)


# ── Task Runner ──────────────────────────────────────────────────────────────


def run_task(task_name: str, episode: int = 0, seed: int = 42) -> float:
    """Run a single task episode and return the grader score."""
    task = TASKS[task_name]
    env = PrintFarmEnv(seed=seed, difficulty=task_name, max_steps=task.max_steps)
    obs = env.reset()
    trajectory = []
    done = False

    # [START] log — exact field names required
    print(f'[START] {json.dumps({"task": task_name, "episode": episode})}')

    step_n = 0
    while not done:
        # Call LLM for action
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": task.description + "\n\n" + LLM_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": json.dumps(obs.model_dump(), indent=2)},
                ],
                temperature=0.0,
                max_tokens=100,
            )
            raw = response.choices[0].message.content.strip()

            # Extract JSON from potential markdown code blocks
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            action_data = json.loads(raw)
            action = Action(**action_data)
        except Exception:
            # Fallback to heuristic on any LLM error
            action = heuristic_action(obs, env.state(), env._step_count)

        obs, reward, done, info = env.step(action)
        trajectory.append(
            {
                "obs": obs.model_dump(),
                "action": action.model_dump(),
                "reward": reward.value,
                "state": env.state(),
            }
        )

        # [STEP] log — exact field names and ordering required
        print(
            f'[STEP] {json.dumps({"step": step_n, "action": action.model_dump(), "reward": reward.value, "done": done})}'
        )
        step_n += 1

    score = task.grader(trajectory)

    # [END] log — exact field names required
    print(f'[END] {json.dumps({"task": task_name, "score": score})}')
    return score


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for name in ["easy", "medium", "hard"]:
        run_task(name, episode=0)
