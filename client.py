"""Print Farm Scheduler Environment Client.

This client uses the OpenEnv WebSocket-based EnvClient to connect to
the print farm scheduler server, enabling efficient multi-step interactions.
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

# Support both in-repo and standalone imports
try:
    from .models import (
        JobSnapshot,
        MachineSnapshot,
        PrintFarmAction,
        PrintFarmObservation,
        RewardBreakdown,
    )
except ImportError:
    from models import (
        JobSnapshot,
        MachineSnapshot,
        PrintFarmAction,
        PrintFarmObservation,
        RewardBreakdown,
    )


class PrintFarmEnv(EnvClient[PrintFarmAction, PrintFarmObservation, State]):
    """
    Client for the Print Farm Scheduler Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> from print_farm_scheduler import PrintFarmEnv, PrintFarmAction
        >>>
        >>> env = PrintFarmEnv(base_url="http://localhost:7860")
        >>> result = env.reset(difficulty="easy", seed=42)
        >>> print(f"Machines: {len(result.observation.machines)}")
        >>>
        >>> # Assign first job to machine 0
        >>> result = env.step(PrintFarmAction(type="assign", machine_id=0))
        >>> print(f"Reward: {result.reward}, Done: {result.done}")
        >>> env.close()

    Example with Docker:
        >>> client = PrintFarmEnv.from_docker_image("print-farm-scheduler:latest")
        >>> try:
        ...     result = client.reset(difficulty="medium", seed=42)
        ...     result = client.step(PrintFarmAction(type="skip"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: PrintFarmAction) -> Dict[str, Any]:
        """
        Convert PrintFarmAction to JSON payload for step message.

        Args:
            action: PrintFarmAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload: Dict[str, Any] = {"type": action.type}
        if action.machine_id is not None:
            payload["machine_id"] = action.machine_id
        if action.job_id is not None:
            payload["job_id"] = action.job_id
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PrintFarmObservation]:
        """
        Parse server response into StepResult[PrintFarmObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with PrintFarmObservation
        """
        obs_data = payload.get("observation", {})

        machines = [
            MachineSnapshot(**m)
            for m in obs_data.get("machines", [])
        ]
        queue = [
            JobSnapshot(**j)
            for j in obs_data.get("queue", [])
        ]
        pending = [
            JobSnapshot(**j)
            for j in obs_data.get("pending_arrivals", [])
        ]
        reward_info = obs_data.get("reward_info")

        observation = PrintFarmObservation(
            step=obs_data.get("step", 0),
            machines=machines,
            queue=queue,
            pending_arrivals=pending,
            completed_count=obs_data.get("completed_count", 0),
            deadlines_missed=obs_data.get("deadlines_missed", 0),
            total_jobs_ever=obs_data.get("total_jobs_ever", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
            reward_info=RewardBreakdown(**reward_info) if reward_info else None,
            rubric_score=obs_data.get("rubric_score"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
