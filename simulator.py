from __future__ import annotations

from copy import deepcopy
import random
from typing import Any


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


DIFFICULTY_PRESETS = {
    "easy": {
        "n_machines": 3,
        "n_jobs": 7,
        "n_arrivals": 0,
        "n_materials": 1,
        "min_ps": 1,
        "max_ps": 3,
        "dl_range": (5, 15),
        "arrival_range": (0, 0),
        "min_fil": 150.0,
        "max_fil": 300.0,
        "speed_range": (0.95, 1.05),
        "weight_cap_range": (90.0, 100.0),
    },
    "medium": {
        "n_machines": 4,
        "n_jobs": 8,
        "n_arrivals": 3,
        "n_materials": 3,
        "min_ps": 1,
        "max_ps": 3,
        "dl_range": (8, 20),
        "arrival_range": (8, 14),
        "min_fil": 100.0,
        "max_fil": 220.0,
        "speed_range": (0.8, 1.2),
        "weight_cap_range": (65.0, 100.0),
    },
    "hard": {
        "n_machines": 6,
        "n_jobs": 14,
        "n_arrivals": 8,
        "n_materials": 5,
        "min_ps": 2,
        "max_ps": 5,
        "dl_range": (4, 12),
        "arrival_range": (4, 20),
        "min_fil": 40.0,
        "max_fil": 120.0,
        "speed_range": (0.6, 1.4),
        "weight_cap_range": (50.0, 100.0),
    },
}


def clamp_task_score(score: float) -> float:
    return max(0.001, min(0.999, float(score)))


def score_difficulty(
    difficulty: str,
    completion_rate: float,
    avg_utilization: float,
    preemption_efficiency: float,
) -> float:
    if difficulty == "easy":
        score = completion_rate
    elif difficulty == "medium":
        score = 0.7 * completion_rate + 0.3 * avg_utilization
    elif difficulty == "hard":
        score = (
            0.7 * completion_rate
            + 0.2 * avg_utilization
            + 0.1 * preemption_efficiency
        )
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    return clamp_task_score(round(score, 4))


class PrintFarmSimulator:
    STEP_DURATION_MINUTES = 20

    def __init__(
        self,
        seed: int | None = None,
        difficulty: str = "easy",
        max_steps: int = 30,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._seed = seed
        self._difficulty = difficulty
        self._max_steps = max_steps
        self._config: dict[str, Any] = {}
        self._rng = random.Random(seed)
        self._step_count = 0
        self._state: dict[str, Any] = {}
        self._utilization_history: list[float] = []
        self.reset(seed=seed, difficulty=difficulty, max_steps=max_steps, **(config or {}))

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def difficulty(self) -> str:
        return self._difficulty

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def reset(
        self,
        *,
        seed: int | None = None,
        difficulty: str | None = None,
        max_steps: int | None = None,
        **config: Any,
    ) -> dict[str, Any]:
        if seed is not None:
            self._seed = seed
        if difficulty is not None:
            self._difficulty = difficulty
        if max_steps is not None:
            self._max_steps = max_steps
        if config:
            self._config.update({k: v for k, v in config.items() if v is not None})

        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._utilization_history = []
        self._state = self._generate_initial_state()
        return self.observation_payload()

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        if not self.is_valid_action(action):
            raise ValueError(f"Illegal action: {action}")

        self._state["failures_this_step"] = 0
        self._state["completions_this_step"] = 0
        self._state["deadlines_missed_this_step"] = 0
        self._state["spool_match_this_step"] = False
        self._state["partial_match_this_step"] = False

        self._apply_action(action)
        self._advance_time()

        self._step_count += 1
        self._utilization_history.append(self._current_utilization())
        done = self._is_terminal()
        reward_info = self._compute_reward(action, done)
        obs = self.observation_payload(
            extra_metadata={
                "progress_signal": reward_info["value"],
                "last_action": deepcopy(action),
            }
        )
        return obs, reward_info, done

    def state_snapshot(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def observation_payload(
        self,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._state
        visible_pending = []
        visible_ahead_steps = int(self._config.get("visible_ahead_steps", VISIBLE_AHEAD_STEPS))
        for job in state["pending_jobs"]:
            if job["arrival_step"] <= self._step_count + visible_ahead_steps:
                visible_pending.append(self._serialize_job(job))

        payload = {
            "step": self._step_count,
            "machines": [self._serialize_machine(machine) for machine in state["machines"]],
            "queue": [self._serialize_job(job) for job in state["queue"]],
            "pending_arrivals": visible_pending,
            "completed_count": len(state["completed"]),
            "deadlines_missed": state["deadlines_missed"],
            "total_jobs_ever": state["total_jobs"],
            "metadata": self._build_metadata(extra_metadata=extra_metadata),
        }
        return payload

    def is_valid_action(self, action: dict[str, Any]) -> bool:
        action_type = action.get("type")
        if action_type == "skip":
            return True

        if action_type == "assign":
            machine_id = action.get("machine_id")
            if machine_id is None:
                return False
            if machine_id < 0 or machine_id >= len(self._state["machines"]):
                return False
            machine = self._state["machines"][machine_id]
            if machine["status"] != "idle":
                return False
            if not self._state["queue"]:
                return False
            job = self._state["queue"][0]
            if not self._material_compatible(machine["material_loaded"], job["material"]):
                return False
            if machine["filament_remaining_g"] < job["weight_g"]:
                return False
            if job["weight_g"] > machine["max_weight_g"]:
                return False
            return True

        if action_type == "preempt":
            machine_id = action.get("machine_id")
            if machine_id is None:
                return False
            if machine_id < 0 or machine_id >= len(self._state["machines"]):
                return False
            return self._state["machines"][machine_id]["status"] in (
                "printing",
                "changing_spool",
            )

        if action_type == "prioritize":
            job_id = action.get("job_id")
            if job_id is None:
                return False
            return any(job["id"] == job_id for job in self._state["queue"])

        return False

    def _effective_preset(self) -> dict[str, Any]:
        preset = deepcopy(DIFFICULTY_PRESETS.get(self._difficulty, DIFFICULTY_PRESETS["easy"]))
        config = self._config

        direct_keys = ("n_machines", "n_jobs", "n_arrivals", "n_materials", "min_ps", "max_ps")
        for key in direct_keys:
            if key in config:
                preset[key] = int(config[key])

        deadline_tightness = float(config.get("deadline_tightness", 1.0))
        min_deadline, max_deadline = preset["dl_range"]
        preset["dl_range"] = (
            max(3, round(min_deadline / max(deadline_tightness, 0.25))),
            max(4, round(max_deadline / max(deadline_tightness, 0.25))),
        )

        arrival_shift = int(config.get("arrival_shift", 0))
        arr_min, arr_max = preset["arrival_range"]
        preset["arrival_range"] = (max(0, arr_min + arrival_shift), max(0, arr_max + arrival_shift))

        filament_scale = float(config.get("filament_scale", 1.0))
        preset["min_fil"] = round(max(5.0, preset["min_fil"] * filament_scale), 2)
        preset["max_fil"] = round(max(preset["min_fil"], preset["max_fil"] * filament_scale), 2)

        speed_scale = float(config.get("speed_scale", 1.0))
        speed_min, speed_max = preset["speed_range"]
        speed_mid = (speed_min + speed_max) / 2.0
        half_span = ((speed_max - speed_min) / 2.0) * max(speed_scale, 0.1)
        preset["speed_range"] = (
            round(max(0.2, speed_mid - half_span), 2),
            round(max(0.25, speed_mid + half_span), 2),
        )

        weight_cap_scale = float(config.get("weight_cap_scale", 1.0))
        cap_min, cap_max = preset["weight_cap_range"]
        preset["weight_cap_range"] = (
            round(max(10.0, cap_min * weight_cap_scale), 1),
            round(max(15.0, cap_max * weight_cap_scale), 1),
        )
        return preset

    def _build_metadata(
        self,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._state
        metrics = self._metrics()
        urgent_jobs = [
            job["id"]
            for job in state["queue"]
            if (job["deadline_step"] - self._step_count) <= job["print_steps"] + 2
        ]
        likely_late_jobs = [
            job["id"]
            for job in state["queue"]
            if (self._step_count + job["print_steps"]) > job["deadline_step"]
        ]
        material_pressure: dict[str, int] = {}
        for job in state["queue"]:
            material_pressure[job["material"]] = material_pressure.get(job["material"], 0) + 1

        score_estimates = {
            difficulty: score_difficulty(
                difficulty,
                metrics["completion_rate"],
                metrics["avg_utilization"],
                metrics["preemption_efficiency"],
            )
            for difficulty in ("easy", "medium", "hard")
        }

        metadata = {
            "difficulty": self._difficulty,
            "remaining_steps": max(0, self._max_steps - self._step_count),
            "current_utilization": round(self._current_utilization(), 4),
            "avg_utilization": round(metrics["avg_utilization"], 4),
            "on_time_completion_rate": round(metrics["completion_rate"], 4),
            "preemption_efficiency": round(metrics["preemption_efficiency"], 4),
            "completed_on_time": metrics["on_time_completions"],
            "queue_size": len(state["queue"]),
            "pending_arrivals": len(state["pending_jobs"]),
            "urgent_job_ids": urgent_jobs,
            "likely_late_job_ids": likely_late_jobs,
            "idle_machine_ids": [machine["id"] for machine in state["machines"] if machine["status"] == "idle"],
            "busy_machine_ids": [machine["id"] for machine in state["machines"] if machine["status"] != "idle"],
            "material_pressure": material_pressure,
            "score_estimates": score_estimates,
            "active_difficulty_score": score_estimates[self._difficulty],
            "curriculum": self.config,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    def _metrics(self) -> dict[str, float | int]:
        total_jobs = max(int(self._state["total_jobs"]), 1)
        on_time_completions = sum(
            1 for completed in self._state["completed"] if completed.get("on_time", False)
        )
        completion_rate = on_time_completions / total_jobs
        if self._state["preemptions"] == 0:
            preemption_efficiency = completion_rate
        else:
            excess_preemptions = max(0, self._state["preemptions"] - on_time_completions)
            penalty = 0.1 * excess_preemptions
            preemption_efficiency = max(0.0, min(1.0, completion_rate * (1.0 - penalty)))
        if self._utilization_history:
            avg_utilization = sum(self._utilization_history) / len(self._utilization_history)
        else:
            avg_utilization = self._current_utilization()
        return {
            "total_jobs": total_jobs,
            "on_time_completions": on_time_completions,
            "completion_rate": completion_rate,
            "avg_utilization": avg_utilization,
            "preemption_efficiency": preemption_efficiency,
        }

    def _serialize_machine(self, machine: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": machine["id"],
            "status": machine["status"],
            "material_loaded": machine["material_loaded"],
            "filament_remaining_g": machine["filament_remaining_g"],
            "speed_modifier": machine["speed_modifier"],
            "max_weight_g": machine["max_weight_g"],
            "hours_used": machine["hours_used"],
            "current_job_id": machine["current_job"]["id"] if machine["current_job"] else None,
            "job_progress_steps": machine["current_job"]["progress"] if machine["current_job"] else 0,
            "job_total_steps": machine["current_job"]["print_steps"] if machine["current_job"] else 0,
        }

    def _serialize_job(self, job: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": job["id"],
            "material": job["material"],
            "weight_g": job["weight_g"],
            "print_steps": job["print_steps"],
            "deadline_step": job["deadline_step"],
            "assigned_machine": job.get("assigned_machine"),
        }

    def _generate_initial_state(self) -> dict[str, Any]:
        preset = self._effective_preset()
        rng = self._rng

        mats = MATERIALS[: max(1, min(int(preset["n_materials"]), len(MATERIALS)))]
        machines = []
        for machine_id in range(max(1, int(preset["n_machines"]))):
            machines.append(
                {
                    "id": machine_id,
                    "status": "idle",
                    "material_loaded": rng.choice(mats),
                    "filament_remaining_g": round(
                        rng.uniform(preset["min_fil"], preset["max_fil"]),
                        1,
                    ),
                    "speed_modifier": round(rng.uniform(*preset["speed_range"]), 2),
                    "max_weight_g": round(rng.uniform(*preset["weight_cap_range"]), 1),
                    "hours_used": 0.0,
                    "current_job": None,
                    "spool_change_remaining": 0,
                }
            )

        job_id = 0
        queue = []
        for _ in range(max(0, int(preset["n_jobs"]))):
            print_steps = rng.randint(int(preset["min_ps"]), int(preset["max_ps"]))
            weight = round(rng.uniform(JOB_MIN_WEIGHT, JOB_MAX_WEIGHT), 1)
            deadline_min, deadline_max = preset["dl_range"]
            deadline = rng.randint(max(print_steps + 2, deadline_min), max(print_steps + 2, deadline_max))
            queue.append(
                {
                    "id": job_id,
                    "material": rng.choice(mats),
                    "weight_g": weight,
                    "print_steps": print_steps,
                    "deadline_step": deadline,
                    "assigned_machine": None,
                }
            )
            job_id += 1

        pending_jobs = []
        for _ in range(max(0, int(preset["n_arrivals"]))):
            print_steps = rng.randint(int(preset["min_ps"]), int(preset["max_ps"]))
            weight = round(rng.uniform(JOB_MIN_WEIGHT, JOB_MAX_WEIGHT), 1)
            arrival_step = rng.randint(*preset["arrival_range"])
            deadline_min, deadline_max = preset["dl_range"]
            deadline = arrival_step + rng.randint(
                max(print_steps + 2, deadline_min),
                max(print_steps + 2, deadline_max),
            )
            pending_jobs.append(
                {
                    "id": job_id,
                    "material": rng.choice(mats),
                    "weight_g": weight,
                    "print_steps": print_steps,
                    "arrival_step": arrival_step,
                    "deadline_step": deadline,
                    "assigned_machine": None,
                }
            )
            job_id += 1

        return {
            "machines": machines,
            "queue": queue,
            "pending_jobs": pending_jobs,
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
            "total_jobs": len(queue) + len(pending_jobs),
        }

    def _apply_action(self, action: dict[str, Any]) -> None:
        action_type = action["type"]
        if action_type == "assign":
            self._do_assign(int(action["machine_id"]))
        elif action_type == "preempt":
            self._do_preempt(int(action["machine_id"]))
        elif action_type == "prioritize":
            self._do_prioritize(int(action["job_id"]))

    def _do_assign(self, machine_id: int) -> None:
        queue = self._state["queue"]
        if not queue:
            return

        machine = self._state["machines"][machine_id]
        if machine["status"] != "idle":
            return

        job = queue.pop(0)
        job["assigned_machine"] = machine["id"]
        compatibility = self._material_compatibility(machine["material_loaded"], job["material"])

        if compatibility == "exact":
            self._state["spool_match_this_step"] = True
        elif compatibility == "partial":
            self._state["partial_match_this_step"] = True
            self._state["spool_changes"] += 1
            job["print_steps"] += PARTIAL_MATCH_EXTRA_STEPS
        else:
            self._state["spool_changes"] += 1

        effective_steps = max(1, round(job["print_steps"] / machine["speed_modifier"]))
        job["print_steps"] = effective_steps
        machine["filament_remaining_g"] -= job["weight_g"]

        if compatibility == "exact":
            machine["status"] = "printing"
            machine["material_loaded"] = job["material"]
            machine["current_job"] = {**job, "progress": 0}
        else:
            machine["status"] = "changing_spool"
            machine["spool_change_remaining"] = SPOOL_CHANGE_STEPS
            machine["material_loaded"] = job["material"]
            machine["current_job"] = {**job, "progress": 0}

    def _do_preempt(self, machine_id: int) -> None:
        machine = self._state["machines"][machine_id]
        if machine["status"] not in ("printing", "changing_spool"):
            return
        if machine["current_job"] is None:
            return

        job = {k: v for k, v in machine["current_job"].items() if k != "progress"}
        job["assigned_machine"] = None
        self._state["queue"].insert(0, job)
        machine["status"] = "idle"
        machine["current_job"] = None
        machine["spool_change_remaining"] = 0
        self._state["preemptions"] += 1

    def _do_prioritize(self, job_id: int) -> None:
        for index, job in enumerate(self._state["queue"]):
            if job["id"] == job_id:
                prioritized = self._state["queue"].pop(index)
                self._state["queue"].insert(0, prioritized)
                break

    def _advance_time(self) -> None:
        step = self._step_count
        degradation_scale = float(self._config.get("degradation_scale", 1.0))
        min_reliability = float(self._config.get("min_reliability", MIN_RELIABILITY))

        for machine in self._state["machines"]:
            if machine["status"] == "changing_spool":
                machine["spool_change_remaining"] -= 1
                if machine["spool_change_remaining"] <= 0:
                    machine["status"] = "printing"
                    machine["spool_change_remaining"] = 0
                continue

            if machine["status"] != "printing" or machine["current_job"] is None:
                continue

            reliability = max(
                min_reliability,
                BASE_RELIABILITY - (DEGRADATION_PER_HOUR * degradation_scale * machine["hours_used"]),
            )

            if self._rng.random() > reliability:
                job = machine["current_job"]
                remaining = max(1, job["print_steps"] - job["progress"])
                failed_job = {
                    "id": job["id"],
                    "material": job["material"],
                    "weight_g": job["weight_g"],
                    "print_steps": remaining,
                    "deadline_step": job["deadline_step"],
                    "assigned_machine": None,
                }
                self._state["queue"].append(failed_job)
                machine["status"] = "idle"
                machine["current_job"] = None
                self._state["failures"] += 1
                self._state["failures_this_step"] += 1
                continue

            machine["current_job"]["progress"] += 1
            machine["hours_used"] += self.STEP_DURATION_MINUTES / 60.0

            if machine["current_job"]["progress"] >= machine["current_job"]["print_steps"]:
                job = machine["current_job"]
                self._state["completed"].append(
                    {
                        "id": job["id"],
                        "completed_at": step + 1,
                        "deadline": job["deadline_step"],
                        "on_time": (step + 1) <= job["deadline_step"],
                    }
                )
                if (step + 1) > job["deadline_step"]:
                    self._state["deadlines_missed"] += 1
                    self._state["deadlines_missed_this_step"] += 1
                self._state["completions_this_step"] += 1
                machine["status"] = "idle"
                machine["current_job"] = None

        remaining_pending = []
        for job in self._state["pending_jobs"]:
            if job["arrival_step"] <= step + 1:
                self._state["queue"].append(job)
            else:
                remaining_pending.append(job)
        self._state["pending_jobs"] = remaining_pending

    def _compute_reward(self, action: dict[str, Any], done: bool) -> dict[str, Any]:
        completions = self._state["completions_this_step"]
        misses = self._state["deadlines_missed_this_step"]
        failures = self._state["failures_this_step"]
        idle_machines = sum(1 for machine in self._state["machines"] if machine["status"] == "idle")

        idle_reward = IDLE_PENALTY * idle_machines
        preempt_reward = PREEMPT_PENALTY if action["type"] == "preempt" else 0.0
        completion_reward = COMPLETION_BONUS * completions
        miss_reward = DEADLINE_MISS_PENALTY * misses
        failure_reward = FAILURE_PENALTY * failures
        skip_reward = SKIP_PENALTY if action["type"] == "skip" else 0.0
        step_reward = STEP_PENALTY
        utilization_reward = MAX_UTILIZATION_BONUS * self._current_utilization()
        spool_reward = SPOOL_MATCH_BONUS if self._state.get("spool_match_this_step", False) else 0.0
        partial_reward = PARTIAL_MATCH_PENALTY if self._state.get("partial_match_this_step", False) else 0.0

        terminal_reward = 0.0
        if done and self._state["total_jobs"] > 0:
            terminal_reward = TERMINAL_BONUS_WEIGHT * (
                sum(1 for completed in self._state["completed"] if completed.get("on_time", False))
                / self._state["total_jobs"]
            )

        raw_reward = (
            idle_reward
            + preempt_reward
            + completion_reward
            + miss_reward
            + failure_reward
            + skip_reward
            + step_reward
            + utilization_reward
            + spool_reward
            + partial_reward
            + terminal_reward
        )

        breakdown = {
            key: round(value, 4)
            for key, value in {
                "idle_penalty": idle_reward,
                "preempt_penalty": preempt_reward,
                "completion_bonus": completion_reward,
                "deadline_miss_penalty": miss_reward,
                "failure_penalty": failure_reward,
                "skip_penalty": skip_reward,
                "step_penalty": step_reward,
                "utilization_bonus": utilization_reward,
                "spool_match_bonus": spool_reward,
                "partial_match_penalty": partial_reward,
                "terminal_bonus": terminal_reward,
            }.items()
            if value != 0
        }

        parts = []
        if completions > 0:
            parts.append(f"{completions} job(s) completed")
        if misses > 0:
            parts.append(f"{misses} deadline(s) missed")
        if failures > 0:
            parts.append(f"{failures} machine failure(s)")
        if terminal_reward > 0:
            parts.append(f"terminal bonus={terminal_reward:.3f}")
        parts.append(f"action={action['type']}")

        return {
            "value": max(0.001, min(0.999, round((raw_reward + 1.0) / 2.0, 4))),
            "breakdown": breakdown,
            "reason": "; ".join(parts),
        }

    def _current_utilization(self) -> float:
        total_machines = len(self._state["machines"])
        if total_machines == 0:
            return 0.0
        active_machines = sum(
            1
            for machine in self._state["machines"]
            if machine["status"] in ("printing", "changing_spool")
        )
        return active_machines / total_machines

    def _material_compatibility(self, loaded: str, required: str) -> str:
        if loaded == required:
            return "exact"
        if MATERIAL_FAMILY.get(loaded) == MATERIAL_FAMILY.get(required):
            return "partial"
        return "incompatible"

    def _material_compatible(self, loaded: str, required: str) -> bool:
        return self._material_compatibility(loaded, required) != "incompatible"

    def _is_terminal(self) -> bool:
        if self._step_count >= self._max_steps:
            return True
        if not self._state["queue"] and not self._state["pending_jobs"]:
            return all(machine["status"] == "idle" for machine in self._state["machines"])
        return False
