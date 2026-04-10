"""
FastAPI application for the Print Farm Scheduler Environment.

This module creates an HTTP server that exposes the PrintFarmEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Or run directly:
    uv run --project . server
"""

import os
import sys
from pathlib import Path

# Ensure imports work in both Docker and local contexts
SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core[core]"
    ) from e

try:
    from ..models import PrintFarmAction, PrintFarmObservation
    from .print_farm_environment import PrintFarmEnvironment
except ImportError:
    from models import PrintFarmAction, PrintFarmObservation
    from server.print_farm_environment import PrintFarmEnvironment


# Create the app using the OpenEnv factory
app = create_app(
    PrintFarmEnvironment,
    PrintFarmAction,
    PrintFarmObservation,
    env_name="print_farm_scheduler",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 7860)
    """
    import uvicorn

    port = int(os.getenv("PORT", str(port)))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
