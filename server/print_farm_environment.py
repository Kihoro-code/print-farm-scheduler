"""
Print Farm Scheduler Environment Implementation.

Integrates the print farm scheduling simulation into the OpenEnv Environment
interface, providing HTTP and WebSocket endpoints for agent interaction.

Each episode simulates a factory shift where an AI agent must optimally assign
print jobs to machines under constraints of material compatibility, filament
capacity, deadlines, mid-episode job arrivals, and wear-based reliability.
"""

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import (
        MachineSnapshot,
        JobSnapshot,
        PrintFarmAction,
        PrintFarmObservation,
        RewardBreakdown,
    )
except ImportError:
    from ..models import (
        MachineSnapshot,
        JobSnapshot,
        PrintFarmAction,
        PrintFarmObservation,
        RewardBreakdown,
    )


# ── Constants ────────────────────────────────────────────────────────────────

MATERIALS = ["PLA", "PETG", "ABS", "Nylon", "TPU"]

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
SPOOL_CHANGE_STEPS = 1
PARTIAL_MATCH_EXTRA_STEPS = 1
BASE_RELIABILITY = 0.95
DEGRADATION_PER_HOUR = 0.02
MIN_RELIABILITY = 0.50

# Difficulty presets
DIFFICULTY_PRESETS = {
    "easy": {
        "n_machines": 3, "n_jobs": 7, "n_arrivals": 0,
        "n_materials": 1, "min_ps": 1, "max_ps": 3,
        "dl_range": (5, 15), "arrival_range": (0, 0),
        "min_fil": 150.0, "max_fil": 300.0,
        "speed_range": (0.95, 1.05), "weight_cap_range": (90.0, 100.0),
    },
    "medium": {
        "n_machines": 4, "n_jobs": 8, "n_arrivals": 3,
        "n_materials": 3, "min_ps": 1, "max_ps": 3,
        "dl_range": (8, 20), "arrival_range": (8, 14),
        "min_fil": 100.0, "max_fil": 220.0,
        "speed_range": (0.8, 1.2), "weight_cap_range": (65.0, 100.0),
    },
    "hard": {
        "n_machines": 6, "n_jobs": 14, "n_arrivals": 8,
        "n_materials": 5, "min_ps": 2, "max_ps": 5,
        "dl_range": (4, 12), "arrival_range": (4, 20),
        "min_fil": 40.0, "max_fil": 120.0,
        "speed_range": (0.6, 1.4), "weight_cap_range": (50.0, 100.0),
    },
}


class PrintFarmEnvironment(Environment):
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

    Design:
    - Multi-step episodes: reset() provides initial factory state, step()
      applies scheduling actions until episode terminates.
    - 30 steps × 20 min/step = 10 hours (one factory shift).
    - Easy tasks ~10 steps, medium ~20, hard needs all 30 with optimal scheduling.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    STEP_DURATION_MINUTES: int = 20

    def __init__(self):
        """Initialize the print farm environment."""
        self._state_obj = State(episode_id=str(uuid4()), step_count=0)
        self._env_state: Dict[str, Any] = {}
        self._rng = random.Random(42)
        self._difficulty = "easy"
        self._max_steps = 30
        self._seed_val: Optional[int] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        max_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> PrintFarmObservation:
        """
        Reset the environment and return initial observation.

        Args:
            seed: Random seed for deterministic episodes.
            episode_id: Optional episode ID.
            difficulty: "easy", "medium", or "hard". Defaults to "easy".
            max_steps: Maximum steps per episode. Defaults to 30.

        Returns:
            PrintFarmObservation with initial factory state.
        """
        if difficulty is not None:
            self._difficulty = difficulty
        if max_steps is not None:
            self._max_steps = max_steps
        if seed is not None:
            self._seed_val = seed

        self._rng = random.Random(self._seed_val)
        self._env_state = self._generate_initial_state()

        self._state_obj = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return self._get_observation()

    def step(
        self,
        action: PrintFarmAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PrintFarmObservation:
        """
        Execute a scheduling action and advance the simulation.

        Args:
            action: PrintFarmAction containing the scheduling decision.
            timeout_s: Optional timeout (unused).

        Returns:
            PrintFarmObservation with updated factory state, reward, and done flag.

        Raises:
            ValueError: If the action is invalid.
        """
        if not self._is_valid_action(action):
            raise ValueError(f"Illegal action: {action}")

        self._env_state["failures_this_step"] = 0
        self._env_state["completions_this_step"] = 0
        self._env_state["deadlines_missed_this_step"] = 0
        self._env_state["spool_match_this_step"] = False
        self._env_state["partial_match_this_step"] = False

        self._apply_action(action)
        self._advance_time()

        self._state_obj.step_count += 1
        done = self._is_terminal()
        reward_info = self._compute_reward(action, done)

        obs = self._get_observation()
        obs.done = done
        obs.reward = reward_info.value
        obs.reward_info = reward_info

        return obs

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state_obj

    # ── Observation ──────────────────────────────────────────────────────

    def _get_observation(self) -> PrintFarmObservation:
        st = self._env_state
        step_count = self._state_obj.step_count

        visible_pending = []
        for j in st["pending_jobs"]:
            if j["arrival_step"] <= step_count + VISIBLE_AHEAD_STEPS:
                visible_pending.append(JobSnapshot(
                    id=j["id"], material=j["material"],
                    weight_g=j["weight_g"], print_steps=j["print_steps"],
                    deadline_step=j["deadline_step"],
                    assigned_machine=j.get("assigned_machine"),
                ))

        machines = []
        for m in st["machines"]:
            machines.append(MachineSnapshot(
                id=m["id"], status=m["status"],
                material_loaded=m["material_loaded"],
                filament_remaining_g=m["filament_remaining_g"],
                speed_modifier=m["speed_modifier"],
                max_weight_g=m["max_weight_g"],
                hours_used=m["hours_used"],
                current_job_id=m["current_job"]["id"] if m["current_job"] else None,
                job_progress_steps=m["current_job"]["progress"] if m["current_job"] else 0,
                job_total_steps=m["current_job"]["print_steps"] if m["current_job"] else 0,
            ))

        queue = [
            JobSnapshot(
                id=j["id"], material=j["material"],
                weight_g=j["weight_g"], print_steps=j["print_steps"],
                deadline_step=j["deadline_step"],
                assigned_machine=j.get("assigned_machine"),
            )
            for j in st["queue"]
        ]

        return PrintFarmObservation(
            step=step_count,
            machines=machines,
            queue=queue,
            pending_arrivals=visible_pending,
            completed_count=len(st["completed"]),
            deadlines_missed=st["deadlines_missed"],
            total_jobs_ever=st["total_jobs"],
            done=False,
            reward=0.0,
        )

    # ── Action Application ───────────────────────────────────────────────

    def _apply_action(self, action: PrintFarmAction) -> None:
        dispatch = {
            "assign": self._do_assign,
            "preempt": self._do_preempt,
            "prioritize": self._do_prioritize,
            "skip": lambda a: None,
        }
        dispatch[action.type](action)

    def _do_assign(self, action: PrintFarmAction) -> None:
        st = self._env_state
        queue = st["queue"]

        if not queue:
            return

        machine = st["machines"][action.machine_id]

        if machine["status"] != "idle":
            return

        job = queue.pop(0)
        job["assigned_machine"] = machine["id"]

        compat = self._material_compatibility(
            machine["material_loaded"], job["material"]
        )

        if compat == "exact":
            st["spool_match_this_step"] = True
        elif compat == "partial":
            st["partial_match_this_step"] = True
            st["spool_changes"] += 1
            job["print_steps"] += PARTIAL_MATCH_EXTRA_STEPS
        else:
            st["spool_match_this_step"] = False
            st["spool_changes"] += 1

        effective_steps = max(1, round(job["print_steps"] / machine["speed_modifier"]))
        job["print_steps"] = effective_steps

        machine["filament_remaining_g"] -= job["weight_g"]

        if compat == "exact":
            machine["status"] = "printing"
            machine["material_loaded"] = job["material"]
            machine["current_job"] = {**job, "progress": 0}
        else:
            machine["status"] = "changing_spool"
            machine["spool_change_remaining"] = SPOOL_CHANGE_STEPS
            machine["material_loaded"] = job["material"]
            machine["current_job"] = {**job, "progress": 0}

    def _do_preempt(self, action: PrintFarmAction) -> None:
        st = self._env_state
        machine = st["machines"][action.machine_id]

        if machine["status"] not in ("printing", "changing_spool"):
            return
        if machine["current_job"] is None:
            return

        job = {k: v for k, v in machine["current_job"].items() if k != "progress"}
        job["assigned_machine"] = None
        st["queue"].insert(0, job)
        machine["status"] = "idle"
        machine["current_job"] = None
        machine["spool_change_remaining"] = 0
        st["preemptions"] += 1

    def _do_prioritize(self, action: PrintFarmAction) -> None:
        queue = self._env_state["queue"]
        for i, job in enumerate(queue):
            if job["id"] == action.job_id:
                prioritized = queue.pop(i)
                queue.insert(0, prioritized)
                break

    # ── Time Advancement ─────────────────────────────────────────────────

    def _advance_time(self) -> None:
        st = self._env_state
        step = self._state_obj.step_count

        for machine in st["machines"]:
            if machine["status"] == "changing_spool":
                machine["spool_change_remaining"] -= 1
                if machine["spool_change_remaining"] <= 0:
                    machine["status"] = "printing"
                    machine["spool_change_remaining"] = 0
                continue

            if machine["status"] != "printing" or machine["current_job"] is None:
                continue

            reliability = max(
                MIN_RELIABILITY,
                BASE_RELIABILITY - DEGRADATION_PER_HOUR * machine["hours_used"],
            )

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
                st["queue"].append(failed_job)
                machine["status"] = "idle"
                machine["current_job"] = None
                st["failures"] += 1
                st["failures_this_step"] += 1
                continue

            machine["current_job"]["progress"] += 1
            machine["hours_used"] += self.STEP_DURATION_MINUTES / 60.0

            if machine["current_job"]["progress"] >= machine["current_job"]["print_steps"]:
                job = machine["current_job"]
                st["completed"].append({
                    "id": job["id"],
                    "completed_at": step + 1,
                    "deadline": job["deadline_step"],
                    "on_time": (step + 1) <= job["deadline_step"],
                })
                if (step + 1) > job["deadline_step"]:
                    st["deadlines_missed"] += 1
                    st["deadlines_missed_this_step"] += 1
                st["completions_this_step"] += 1
                machine["status"] = "idle"
                machine["current_job"] = None

        still_pending = []
        for j in st["pending_jobs"]:
            if j["arrival_step"] <= step + 1:
                st["queue"].append(j)
            else:
                still_pending.append(j)
        st["pending_jobs"] = still_pending

    # ── Reward Computation ───────────────────────────────────────────────

    def _compute_reward(self, action: PrintFarmAction, done: bool) -> RewardBreakdown:
        st = self._env_state

        comps = st["completions_this_step"]
        misses = st["deadlines_missed_this_step"]
        fails = st["failures_this_step"]
        idle = sum(1 for m in st["machines"] if m["status"] == "idle")

        idle_r = IDLE_PENALTY * idle
        pre_r = PREEMPT_PENALTY if action.type == "preempt" else 0.0
        comp_r = COMPLETION_BONUS * comps
        miss_r = DEADLINE_MISS_PENALTY * misses
        fail_r = FAILURE_PENALTY * fails
        skip_r = SKIP_PENALTY if action.type == "skip" else 0.0
        step_r = STEP_PENALTY

        total = len(st["machines"])
        util = (total - idle) / total if total > 0 else 0.0
        util_r = MAX_UTILIZATION_BONUS * util

        spool_r = SPOOL_MATCH_BONUS if st.get("spool_match_this_step", False) else 0.0
        partial_r = PARTIAL_MATCH_PENALTY if st.get("partial_match_this_step", False) else 0.0

        terminal_r = 0.0
        if done:
            total_jobs = st["total_jobs"]
            if total_jobs > 0:
                on_time = sum(1 for c in st["completed"] if c.get("on_time", False))
                terminal_r = TERMINAL_BONUS_WEIGHT * (on_time / total_jobs)

        raw = (idle_r + pre_r + comp_r + miss_r + fail_r + skip_r +
               step_r + util_r + spool_r + partial_r + terminal_r)

        breakdown = {}
        for name, val in [
            ("idle_penalty", idle_r), ("preempt_penalty", pre_r),
            ("completion_bonus", comp_r), ("deadline_miss_penalty", miss_r),
            ("failure_penalty", fail_r), ("skip_penalty", skip_r),
            ("step_penalty", step_r), ("utilization_bonus", util_r),
            ("spool_match_bonus", spool_r), ("partial_match_penalty", partial_r),
            ("terminal_bonus", terminal_r),
        ]:
            if val != 0:
                breakdown[name] = round(val, 4)

        value = max(-1.0, min(1.0, round(raw, 4)))

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

        return RewardBreakdown(value=value, breakdown=breakdown, reason=reason)

    # ── Material Compatibility ───────────────────────────────────────────

    def _material_compatibility(self, loaded: str, required: str) -> str:
        if loaded == required:
            return "exact"
        if MATERIAL_FAMILY.get(loaded) == MATERIAL_FAMILY.get(required):
            return "partial"
        return "incompatible"

    def _material_compatible(self, loaded: str, required: str) -> bool:
        return self._material_compatibility(loaded, required) != "incompatible"

    # ── Validation ───────────────────────────────────────────────────────

    def _is_valid_action(self, action: PrintFarmAction) -> bool:
        if action.type == "skip":
            return True

        if action.type == "assign":
            if action.machine_id is None:
                return False
            if action.machine_id < 0 or action.machine_id >= len(self._env_state["machines"]):
                return False
            machine = self._env_state["machines"][action.machine_id]
            if machine["status"] != "idle":
                return False
            if not self._env_state["queue"]:
                return False
            job = self._env_state["queue"][0]
            if not self._material_compatible(machine["material_loaded"], job["material"]):
                return False
            if machine["filament_remaining_g"] < job["weight_g"]:
                return False
            if job["weight_g"] > machine["max_weight_g"]:
                return False
            return True

        if action.type == "preempt":
            if action.machine_id is None:
                return False
            if action.machine_id < 0 or action.machine_id >= len(self._env_state["machines"]):
                return False
            return self._env_state["machines"][action.machine_id]["status"] in (
                "printing", "changing_spool",
            )

        if action.type == "prioritize":
            if action.job_id is None:
                return False
            return any(j["id"] == action.job_id for j in self._env_state["queue"])

        return False

    def _is_terminal(self) -> bool:
        if self._state_obj.step_count >= self._max_steps:
            return True
        st = self._env_state
        if not st["queue"] and not st["pending_jobs"]:
            if all(m["status"] == "idle" for m in st["machines"]):
                return True
        return False

    # ── State Generation ─────────────────────────────────────────────────

    def _generate_initial_state(self) -> Dict[str, Any]:
        preset = DIFFICULTY_PRESETS.get(self._difficulty, DIFFICULTY_PRESETS["easy"])
        rng = self._rng

        n_machines = preset["n_machines"]
        n_jobs = preset["n_jobs"]
        n_arrivals = preset["n_arrivals"]
        mats = MATERIALS[:preset["n_materials"]]

        machines = []
        for i in range(n_machines):
            machines.append({
                "id": i,
                "status": "idle",
                "material_loaded": rng.choice(mats),
                "filament_remaining_g": round(rng.uniform(preset["min_fil"], preset["max_fil"]), 1),
                "speed_modifier": round(rng.uniform(*preset["speed_range"]), 2),
                "max_weight_g": round(rng.uniform(*preset["weight_cap_range"]), 1),
                "hours_used": 0.0,
                "current_job": None,
                "spool_change_remaining": 0,
            })

        job_id = 0
        jobs = []
        for _ in range(n_jobs):
            ps = rng.randint(preset["min_ps"], preset["max_ps"])
            weight = round(rng.uniform(JOB_MIN_WEIGHT, JOB_MAX_WEIGHT), 1)
            deadline = rng.randint(ps + 2, ps + rng.randint(3, MAX_DEADLINE_SLACK))
            jobs.append({
                "id": job_id,
                "material": rng.choice(mats),
                "weight_g": weight,
                "print_steps": ps,
                "deadline_step": deadline,
                "assigned_machine": None,
            })
            job_id += 1

        pending = []
        for _ in range(n_arrivals):
            ps = rng.randint(preset["min_ps"], preset["max_ps"])
            weight = round(rng.uniform(JOB_MIN_WEIGHT, JOB_MAX_WEIGHT), 1)
            arr = rng.randint(*preset["arrival_range"])
            deadline = arr + rng.randint(ps + 2, ps + rng.randint(3, MAX_DEADLINE_SLACK))
            pending.append({
                "id": job_id,
                "material": rng.choice(mats),
                "weight_g": weight,
                "print_steps": ps,
                "arrival_step": arr,
                "deadline_step": deadline,
                "assigned_machine": None,
            })
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
