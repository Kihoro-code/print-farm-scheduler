from pydantic import BaseModel, Field
from typing import Literal
import random


# ── Pydantic Models (OpenEnv spec requirement) ──────────────────────────────


class MachineSnapshot(BaseModel):
    id: int
    status: Literal["idle", "printing", "changing_spool"]
    material_loaded: str
    filament_remaining_g: float
    speed_modifier: float = 1.0
    max_weight_g: float = 100.0
    hours_used: float = 0.0
    current_job_id: int | None = None
    job_progress_steps: int = 0
    job_total_steps: int = 0


class JobSnapshot(BaseModel):
    id: int
    material: str
    weight_g: float
    print_steps: int
    deadline_step: int
    assigned_machine: int | None = None


class Observation(BaseModel):
    step: int
    machines: list[MachineSnapshot]
    queue: list[JobSnapshot]
    pending_arrivals: list[JobSnapshot]  # Advance customer bookings visible ≤2 steps ahead
    completed_count: int
    deadlines_missed: int
    total_jobs_ever: int


class Action(BaseModel):
    type: Literal["assign", "preempt", "prioritize", "skip"]
    machine_id: int | None = None
    job_id: int | None = None


class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    breakdown: dict[str, float]
    reason: str


# ── Constants ────────────────────────────────────────────────────────────────

MATERIALS = ["PLA", "PETG", "ABS", "Nylon", "TPU"]

# Material compatibility families — materials in the same family can be
# partially compatible (printable but with a speed/quality penalty).
MATERIAL_FAMILY = {
    "PLA": "low_temp",
    "PETG": "low_temp",
    "ABS": "mid_temp",
    "Nylon": "specialty",
    "TPU": "specialty",
}

IDLE_PENALTY = -0.01
PREEMPT_PENALTY = -0.03
SKIP_PENALTY = -0.005
STEP_PENALTY = -0.001
COMPLETION_BONUS = 0.20
DEADLINE_MISS_PENALTY = -0.10
SPOOL_CHANGE_PENALTY = -0.01
FAILURE_PENALTY = -0.02
MAX_UTILIZATION_BONUS = 0.05
SPOOL_MATCH_BONUS = 0.02
PARTIAL_MATCH_PENALTY = -0.015
TERMINAL_BONUS_WEIGHT = 0.3
MIN_FILAMENT_G = 50.0
MAX_FILAMENT_G = 200.0
MIN_PRINT_STEPS = 1
MAX_DEADLINE_SLACK = 12
JOB_MIN_WEIGHT = 10.0
JOB_MAX_WEIGHT = 60.0
VISIBLE_AHEAD_STEPS = 2

# Spool change cooldown (steps a machine spends swapping material)
SPOOL_CHANGE_STEPS = 1

# Partial material-family match adds extra print steps
PARTIAL_MATCH_EXTRA_STEPS = 1

# Wear-based machine degradation
BASE_RELIABILITY = 0.95
DEGRADATION_PER_HOUR = 0.02
MIN_RELIABILITY = 0.50


# ── Environment Class ────────────────────────────────────────────────────────


class PrintFarmEnv:
    """
    Enterprise 3D Print Farm Logistics Scheduler.

    Simulates an industrial 3D printing facility where an AI agent must
    optimally assign print jobs to a fleet of machines under constraints
    including material compatibility, filament capacity, deadlines,
    mid-episode job arrivals, and wear-based machine reliability.

    Key mechanics:
    - Material families: PLA↔PETG and Nylon↔TPU are partially compatible
      (printable with a quality penalty), all others require exact match.
    - Spool change cooldown: switching materials costs 1 step of downtime.
    - Machine differentiation: machines vary in speed and max build weight.
    - Wear degradation: reliability drops with cumulative usage hours.
    - Terminal reward: episode-end bonus based on overall completion rate.
    """

    # 30 steps × 20 min/step = 10 hours — one factory shift.
    # Episode length calibrated so easy tasks are solvable in ~10 steps,
    # medium in ~20, and hard needs all 30 with optimal scheduling.
    STEP_DURATION_MINUTES: int = 20

    def __init__(
        self, seed: int | None = None, difficulty: str = "easy", max_steps: int = 30
    ):
        self._seed = seed
        self._rng = random.Random(seed)
        self._difficulty = difficulty
        self.MAX_STEPS = max_steps
        self._step_count: int = 0
        self._state: dict = {}
        self.reset()

    def reset(self) -> Observation:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._state = self._generate_initial_state()
        return self._get_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if not self._is_valid_action(action):
            raise ValueError(f"Illegal action: {action}")

        self._state["failures_this_step"] = 0
        self._state["completions_this_step"] = 0
        self._state["deadlines_missed_this_step"] = 0
        self._state["spool_match_this_step"] = False
        self._state["partial_match_this_step"] = False

        self._apply_action(action)
        self._advance_time()

        self._step_count += 1
        obs = self._get_observation()
        done = self._is_terminal()
        reward = self._compute_reward(action, done)
        info = {
            "step": self._step_count,
            "env_id": "print-farm-scheduler",
            "difficulty": self._difficulty,
        }
        return obs, reward, done, info

    def state(self) -> dict:
        return self._state.copy()

    # ── Observation ──────────────────────────────────────────────────────

    def _get_observation(self) -> Observation:
        state = self._state

        visible_pending = []
        for j in state["pending_jobs"]:
            if j["arrival_step"] <= self._step_count + VISIBLE_AHEAD_STEPS:
                visible_pending.append(JobSnapshot(**j))

        machines = [
            MachineSnapshot(
                id=m["id"],
                status=m["status"],
                material_loaded=m["material_loaded"],
                filament_remaining_g=m["filament_remaining_g"],
                speed_modifier=m["speed_modifier"],
                max_weight_g=m["max_weight_g"],
                hours_used=m["hours_used"],
                current_job_id=m["current_job"]["id"] if m["current_job"] else None,
                job_progress_steps=m["current_job"]["progress"]
                if m["current_job"]
                else 0,
                job_total_steps=m["current_job"]["print_steps"]
                if m["current_job"]
                else 0,
            )
            for m in state["machines"]
        ]

        queue = [JobSnapshot(**j) for j in state["queue"]]

        return Observation(
            step=self._step_count,
            machines=machines,
            queue=queue,
            pending_arrivals=visible_pending,
            completed_count=len(state["completed"]),
            deadlines_missed=state["deadlines_missed"],
            total_jobs_ever=state["total_jobs"],
        )

    # ── Action Application ───────────────────────────────────────────────

    def _apply_action(self, action: Action) -> None:
        dispatch = {
            "assign": self._do_assign,
            "preempt": self._do_preempt,
            "prioritize": self._do_prioritize,
            "skip": lambda a: None,
        }
        dispatch[action.type](action)

    def _do_assign(self, action: Action) -> None:
        state = self._state
        queue = state["queue"]

        if not queue:
            return

        machine = state["machines"][action.machine_id]

        if machine["status"] != "idle":
            return

        job = queue.pop(0)
        job["assigned_machine"] = machine["id"]

        # Determine material compatibility level
        compat = self._material_compatibility(
            machine["material_loaded"], job["material"]
        )

        if compat == "exact":
            state["spool_match_this_step"] = True
        elif compat == "partial":
            # Partial family match: can print but with penalty
            state["partial_match_this_step"] = True
            state["spool_changes"] += 1
            # Add extra print steps for quality compensation
            job["print_steps"] += PARTIAL_MATCH_EXTRA_STEPS
        else:
            # Should not reach here due to validation, but safety fallback
            state["spool_match_this_step"] = False
            state["spool_changes"] += 1

        # Apply machine speed modifier to effective print steps
        effective_steps = max(1, round(job["print_steps"] / machine["speed_modifier"]))
        job["print_steps"] = effective_steps

        # Consume filament
        machine["filament_remaining_g"] -= job["weight_g"]

        # Determine if spool change cooldown is needed
        if compat == "exact":
            # Exact match — start printing immediately
            machine["status"] = "printing"
            machine["material_loaded"] = job["material"]
            machine["current_job"] = {
                **job,
                "progress": 0,
            }
        else:
            # Partial or family match — need to change spool first
            machine["status"] = "changing_spool"
            machine["spool_change_remaining"] = SPOOL_CHANGE_STEPS
            machine["material_loaded"] = job["material"]
            machine["current_job"] = {
                **job,
                "progress": 0,
            }

    def _do_preempt(self, action: Action) -> None:
        state = self._state
        machine = state["machines"][action.machine_id]

        if machine["status"] not in ("printing", "changing_spool"):
            return
        if machine["current_job"] is None:
            return

        job = {k: v for k, v in machine["current_job"].items() if k != "progress"}
        job["assigned_machine"] = None
        state["queue"].insert(0, job)
        machine["status"] = "idle"
        machine["current_job"] = None
        machine["spool_change_remaining"] = 0
        state["preemptions"] += 1

    def _do_prioritize(self, action: Action) -> None:
        queue = self._state["queue"]
        for i, job in enumerate(queue):
            if job["id"] == action.job_id:
                prioritized = queue.pop(i)
                queue.insert(0, prioritized)
                break

    # ── Time Advancement ─────────────────────────────────────────────────

    def _advance_time(self) -> None:
        state = self._state
        step = self._step_count

        for machine in state["machines"]:
            # Handle spool change cooldown
            if machine["status"] == "changing_spool":
                machine["spool_change_remaining"] -= 1
                if machine["spool_change_remaining"] <= 0:
                    machine["status"] = "printing"
                    machine["spool_change_remaining"] = 0
                continue

            if machine["status"] != "printing" or machine["current_job"] is None:
                continue

            # Wear-based reliability: degrades with cumulative usage
            reliability = max(
                MIN_RELIABILITY,
                BASE_RELIABILITY - DEGRADATION_PER_HOUR * machine["hours_used"],
            )

            # Machine reliability check (wear-based)
            if self._rng.random() > reliability:
                job = machine["current_job"]
                remaining = job["print_steps"] - job["progress"]
                failed_job = {
                    "id": job["id"],
                    "material": job["material"],
                    "weight_g": job["weight_g"],
                    "print_steps": remaining,
                    "deadline_step": job["deadline_step"],
                    "assigned_machine": None,
                }
                state["queue"].append(failed_job)
                machine["status"] = "idle"
                machine["current_job"] = None
                state["failures"] += 1
                state["failures_this_step"] += 1
                continue

            # Advance progress and accumulate wear
            machine["current_job"]["progress"] += 1
            machine["hours_used"] += self.STEP_DURATION_MINUTES / 60.0

            # Check completion
            if (
                machine["current_job"]["progress"]
                >= machine["current_job"]["print_steps"]
            ):
                job = machine["current_job"]
                state["completed"].append(
                    {
                        "id": job["id"],
                        "completed_at": step + 1,
                        "deadline": job["deadline_step"],
                        "on_time": (step + 1) <= job["deadline_step"],
                    }
                )
                if (step + 1) > job["deadline_step"]:
                    state["deadlines_missed"] += 1
                    state["deadlines_missed_this_step"] += 1
                state["completions_this_step"] += 1
                machine["status"] = "idle"
                machine["current_job"] = None

        # Spawn pending jobs whose arrival time has come
        still_pending = []
        for j in state["pending_jobs"]:
            if j["arrival_step"] <= step + 1:
                state["queue"].append(j)
            else:
                still_pending.append(j)
        state["pending_jobs"] = still_pending

    # ── Reward Computation ───────────────────────────────────────────────

    def _compute_reward(self, action: Action, done: bool) -> Reward:
        state = self._state

        comps = state["completions_this_step"]
        misses = state["deadlines_missed_this_step"]
        fails = state["failures_this_step"]
        idle = sum(1 for m in state["machines"] if m["status"] == "idle")

        idle_r = IDLE_PENALTY * idle
        pre_r = PREEMPT_PENALTY if action.type == "preempt" else 0.0
        comp_r = COMPLETION_BONUS * comps
        miss_r = DEADLINE_MISS_PENALTY * misses
        fail_r = FAILURE_PENALTY * fails
        skip_r = SKIP_PENALTY if action.type == "skip" else 0.0
        step_r = STEP_PENALTY

        # Utilization bonus
        total = len(state["machines"])
        util = (total - idle) / total if total > 0 else 0.0
        util_r = MAX_UTILIZATION_BONUS * util

        # Spool-match bonus for assigning without material switch
        spool_r = (
            SPOOL_MATCH_BONUS if state.get("spool_match_this_step", False) else 0.0
        )

        # Partial match penalty (family-compatible but not exact)
        partial_r = (
            PARTIAL_MATCH_PENALTY
            if state.get("partial_match_this_step", False)
            else 0.0
        )

        # Terminal episode reward — completion-rate-based bonus
        terminal_r = 0.0
        if done:
            total_jobs = state["total_jobs"]
            if total_jobs > 0:
                on_time = sum(
                    1 for c in state["completed"] if c.get("on_time", False)
                )
                terminal_r = TERMINAL_BONUS_WEIGHT * (on_time / total_jobs)

        raw = (
            idle_r
            + pre_r
            + comp_r
            + miss_r
            + fail_r
            + skip_r
            + step_r
            + util_r
            + spool_r
            + partial_r
            + terminal_r
        )

        breakdown = {}
        if idle_r != 0:
            breakdown["idle_penalty"] = round(idle_r, 4)
        if pre_r != 0:
            breakdown["preempt_penalty"] = round(pre_r, 4)
        if comp_r != 0:
            breakdown["completion_bonus"] = round(comp_r, 4)
        if miss_r != 0:
            breakdown["deadline_miss_penalty"] = round(miss_r, 4)
        if fail_r != 0:
            breakdown["failure_penalty"] = round(fail_r, 4)
        if skip_r != 0:
            breakdown["skip_penalty"] = round(skip_r, 4)
        if step_r != 0:
            breakdown["step_penalty"] = round(step_r, 4)
        if util_r != 0:
            breakdown["utilization_bonus"] = round(util_r, 4)
        if spool_r != 0:
            breakdown["spool_match_bonus"] = round(spool_r, 4)
        if partial_r != 0:
            breakdown["partial_match_penalty"] = round(partial_r, 4)
        if terminal_r != 0:
            breakdown["terminal_bonus"] = round(terminal_r, 4)

        value = max(0.001, min(0.999, round((raw + 1.0) / 2.0, 4)))

        parts = []
        if comps > 0:
            parts.append(f"{comps} job(s) completed")
        if misses > 0:
            parts.append(f"{misses} deadline(s) missed")
        if fails > 0:
            parts.append(f"{fails} machine failure(s)")
        if terminal_r > 0:
            parts.append(f"terminal bonus={terminal_r:.3f}")
        parts.append(f"action={action.type}")
        reason = "; ".join(parts) if parts else f"action={action.type}"

        return Reward(value=value, breakdown=breakdown, reason=reason)

    # ── Material Compatibility ───────────────────────────────────────────

    def _material_compatibility(self, loaded: str, required: str) -> str:
        """
        Determine the compatibility level between loaded and required material.

        Returns:
            "exact"   — same material, no penalty
            "partial" — same family (e.g. PLA↔PETG), printable with penalty
            "incompatible" — different families, cannot print
        """
        if loaded == required:
            return "exact"
        if MATERIAL_FAMILY.get(loaded) == MATERIAL_FAMILY.get(required):
            return "partial"
        return "incompatible"

    def _material_compatible(self, loaded: str, required: str) -> bool:
        """Check if a machine can accept a job (exact or partial match)."""
        return self._material_compatibility(loaded, required) != "incompatible"

    # ── Validation ───────────────────────────────────────────────────────

    def _is_valid_action(self, action: Action) -> bool:
        if action.type == "skip":
            return True

        if action.type == "assign":
            if action.machine_id is None:
                return False
            if action.machine_id < 0 or action.machine_id >= len(
                self._state["machines"]
            ):
                return False
            machine = self._state["machines"][action.machine_id]
            if machine["status"] != "idle":
                return False
            if not self._state["queue"]:
                return False
            job = self._state["queue"][0]
            if not self._material_compatible(
                machine["material_loaded"], job["material"]
            ):
                return False
            if machine["filament_remaining_g"] < job["weight_g"]:
                return False
            if job["weight_g"] > machine["max_weight_g"]:
                return False
            return True

        if action.type == "preempt":
            if action.machine_id is None:
                return False
            if action.machine_id < 0 or action.machine_id >= len(
                self._state["machines"]
            ):
                return False
            return self._state["machines"][action.machine_id]["status"] in (
                "printing",
                "changing_spool",
            )

        if action.type == "prioritize":
            if action.job_id is None:
                return False
            return any(j["id"] == action.job_id for j in self._state["queue"])

        return False

    def _is_terminal(self) -> bool:
        if self._step_count >= self.MAX_STEPS:
            return True
        state = self._state
        if not state["queue"] and not state["pending_jobs"]:
            if all(m["status"] == "idle" for m in state["machines"]):
                return True
        return False

    # ── State Generation ─────────────────────────────────────────────────

    def _generate_initial_state(self) -> dict:
        diff = self._difficulty
        rng = self._rng

        if diff == "easy":
            n_machines, n_jobs, n_arrivals = 3, 7, 0
            n_materials = 1
            min_ps, max_ps = 1, 3
            dl_range = (5, 15)
            arrival_range = (0, 0)
            min_fil, max_fil = 150.0, 300.0
            speed_range = (0.95, 1.05)
            weight_cap_range = (90.0, 100.0)
        elif diff == "medium":
            n_machines, n_jobs, n_arrivals = 4, 8, 3
            n_materials = 3
            min_ps, max_ps = 1, 3
            dl_range = (8, 20)
            arrival_range = (8, 14)
            min_fil, max_fil = 100.0, 220.0
            speed_range = (0.8, 1.2)
            weight_cap_range = (65.0, 100.0)
        else:
            n_machines, n_jobs, n_arrivals = 6, 14, 8
            n_materials = 5
            min_ps, max_ps = 2, 5
            dl_range = (4, 12)
            arrival_range = (4, 20)
            min_fil, max_fil = 40.0, 120.0
            speed_range = (0.6, 1.4)
            weight_cap_range = (50.0, 100.0)

        mats = MATERIALS[:n_materials]

        machines = []
        for i in range(n_machines):
            machines.append(
                {
                    "id": i,
                    "status": "idle",
                    "material_loaded": rng.choice(mats),
                    "filament_remaining_g": round(rng.uniform(min_fil, max_fil), 1),
                    "speed_modifier": round(rng.uniform(*speed_range), 2),
                    "max_weight_g": round(rng.uniform(*weight_cap_range), 1),
                    "hours_used": 0.0,
                    "current_job": None,
                    "spool_change_remaining": 0,
                }
            )

        job_id = 0
        jobs = []
        for _ in range(n_jobs):
            ps = rng.randint(min_ps, max_ps)
            weight = round(rng.uniform(JOB_MIN_WEIGHT, JOB_MAX_WEIGHT), 1)
            deadline = self._step_count + rng.randint(
                ps + 2, ps + rng.randint(3, MAX_DEADLINE_SLACK)
            )
            jobs.append(
                {
                    "id": job_id,
                    "material": rng.choice(mats),
                    "weight_g": weight,
                    "print_steps": ps,
                    "deadline_step": deadline,
                    "assigned_machine": None,
                }
            )
            job_id += 1

        pending = []
        for _ in range(n_arrivals):
            ps = rng.randint(min_ps, max_ps)
            weight = round(rng.uniform(JOB_MIN_WEIGHT, JOB_MAX_WEIGHT), 1)
            arr = rng.randint(*arrival_range)
            deadline = arr + rng.randint(
                ps + 2, ps + rng.randint(3, MAX_DEADLINE_SLACK)
            )
            pending.append(
                {
                    "id": job_id,
                    "material": rng.choice(mats),
                    "weight_g": weight,
                    "print_steps": ps,
                    "arrival_step": arr,
                    "deadline_step": deadline,
                    "assigned_machine": None,
                }
            )
            job_id += 1

        return {
            "machines": machines,
            "queue": list(jobs),
            "pending_jobs": pending,
            "completed": [],
            "deadlines_missed": 0,
            "preemptions": 0,
            "spool_changes": 0,
            "failures": 0,
            "failures_this_step": 0,
            "completions_this_step": 0,
            "deadlines_missed_this_step": 0,
            "spool_match_this_step": False,
            "partial_match_this_step": False,
            "total_jobs": n_jobs + n_arrivals,
        }
