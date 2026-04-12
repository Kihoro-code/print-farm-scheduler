from dataclasses import dataclass
from typing import Callable

from simulator import clamp_task_score, score_difficulty


@dataclass
class Task:
    name: str
    difficulty: str
    description: str
    grader: Callable
    max_steps: int
    seed: int = 42


# ── Helpers ─────────────────────────────────────────────────────────────────


def _clamp_score(score: float) -> float:
    """Clamp score to strictly between 0 and 1 (validator rejects 0.0 and 1.0)."""
    return clamp_task_score(score)


# ── Graders ─────────────────────────────────────────────────────────────────


def grade_easy(trajectory: list[dict]) -> float:
    """
    Easy task: % of jobs completed before deadline.
    All jobs are available at start, same material, generous deadlines.
    """
    completions = _count_completions(trajectory)
    total = _total_jobs(trajectory)
    if total == 0:
        return 0.001
    return score_difficulty("easy", completions / total, 0.0, 0.0)


def grade_medium(trajectory: list[dict]) -> float:
    """
    Medium task: 70% completion rate + 30% utilization.
    Mixed materials, rolling arrivals, tight deadlines on some jobs.
    """
    completions = _count_completions(trajectory)
    total = _total_jobs(trajectory)
    utilization = _avg_utilization(trajectory)

    if total == 0:
        return 0.001

    completion_rate = completions / total
    return score_difficulty("medium", completion_rate, utilization, completion_rate)


def grade_hard(trajectory: list[dict]) -> float:
    """
    Hard task: 70% completion + 20% utilization + 10% low-preemption bonus.
    Many materials, tight deadlines, wave arrivals, filament scarcity.
    """
    completions = _count_completions(trajectory)
    total = _total_jobs(trajectory)
    utilization = _avg_utilization(trajectory)
    preemption_efficiency = _preemption_efficiency(trajectory)

    if total == 0:
        return 0.001

    completion_rate = completions / total
    return score_difficulty(
        "hard",
        completion_rate,
        utilization,
        preemption_efficiency,
    )


# ── Grader Helpers ──────────────────────────────────────────────────────────


def _count_completions(trajectory: list[dict]) -> int:
    """Count jobs completed before deadline from final trajectory state."""
    if not trajectory:
        return 0
    final_state = trajectory[-1].get("state", {})
    completed = final_state.get("completed", [])
    return sum(1 for c in completed if c.get("on_time", False))


def _total_jobs(trajectory: list[dict]) -> int:
    """Total jobs that existed (completed + still pending/queued + printing)."""
    if not trajectory:
        return 0
    final_state = trajectory[-1].get("state", {})
    completed = len(final_state.get("completed", []))
    queue = len(final_state.get("queue", []))
    pending = len(final_state.get("pending_jobs", []))
    printing = sum(
        1 for m in final_state.get("machines", []) if m.get("current_job") is not None
    )
    return completed + queue + pending + printing


def _avg_utilization(trajectory: list[dict]) -> float:
    """Average machine utilization across the episode (printing + changing_spool)."""
    if not trajectory:
        return 0.0
    total_util = 0.0
    for step_data in trajectory:
        state = step_data.get("state", {})
        machines = state.get("machines", [])
        if not machines:
            continue
        active = sum(
            1
            for m in machines
            if m.get("status") in ("printing", "changing_spool")
        )
        total_util += active / len(machines)
    return round(total_util / len(trajectory), 4) if trajectory else 0.0


def _preemption_efficiency(trajectory: list[dict]) -> float:
    """
    Score preemption effectiveness. Rewards preemptions that contributed to
    on-time completions, penalizes excessive preemptions that didn't help.
    """
    if not trajectory:
        return 0.0
    final_state = trajectory[-1].get("state", {})
    preemptions = final_state.get("preemptions", 0)
    completions = sum(
        1 for c in final_state.get("completed", []) if c.get("on_time", False)
    )
    total = final_state.get("total_jobs", 1)

    # Completion rate as baseline
    completion_rate = completions / max(total, 1)

    if preemptions == 0:
        # No preemptions — score by completion rate alone
        return round(completion_rate, 4)

    # Preemptions should help, not hurt — penalize excess preemptions
    # that didn't translate to completions
    excess_preemptions = max(0, preemptions - completions)
    penalty = 0.1 * excess_preemptions
    efficiency = max(0.0, min(1.0, completion_rate * (1.0 - penalty)))
    return round(efficiency, 4)


# ── Task Definitions ────────────────────────────────────────────────────────

TASKS: dict[str, Task] = {
    "easy": Task(
        name="easy",
        difficulty="easy",
        description=(
            "You manage a small 3D print farm with 3 machines. All jobs use PLA. "
            "Assign jobs to idle machines to complete as many as possible before "
            "their deadlines. Jobs are available immediately with generous deadlines. "
            "Focus on keeping machines busy — there are no material conflicts. "
            "Machines differ slightly in print speed (speed_modifier). "
            "Faster machines complete jobs in fewer steps."
        ),
        grader=grade_easy,
        max_steps=30,
        seed=42,
    ),
    "medium": Task(
        name="medium",
        difficulty="medium",
        description=(
            "You manage a mid-size print farm with 4 machines and 3 material types "
            "(PLA, PETG, ABS). PLA and PETG are in the same material family — "
            "a PLA machine CAN print PETG jobs (and vice versa) but with +1 extra "
            "print step and a 1-step spool change delay. Exact material matches "
            "start printing immediately with no penalty. Jobs arrive in two waves — "
            "some at start, more around step 10. Machines differ in speed and max "
            "build weight (max_weight_g). Match materials carefully, watch filament "
            "levels, and prioritize tight-deadline jobs. Preempt if a machine is "
            "blocked on a low-priority job when an urgent one needs assignment."
        ),
        grader=grade_medium,
        max_steps=30,
        seed=42,
    ),
    "hard": Task(
        name="hard",
        difficulty="hard",
        description=(
            "You manage a large print farm with 6 machines and 5 material types "
            "(PLA, PETG, ABS, Nylon, TPU). Material families: PLA↔PETG (low-temp) "
            "and Nylon↔TPU (specialty) are partially compatible — printable with "
            "+1 step penalty and a 1-step spool change delay. ABS requires exact "
            "match. 14 jobs start in queue with 8 more arriving in waves with very "
            "tight deadlines. Filament is scarce (40–120g per machine), spool changes "
            "are costly, and machines vary in speed (speed_modifier 0.6–1.4×) and "
            "max build weight (max_weight_g 50–100g). Machines degrade with use — "
            "reliability drops as hours_used increases (starts at 95%, decreasing "
            "~2% per hour of printing, minimum 50%). You must preempt strategically, "
            "prioritize the queue, balance load across machines to minimize wear, "
            "and match materials efficiently to maximize on-time completions."
        ),
        grader=grade_hard,
        max_steps=30,
        seed=42,
    ),
}
