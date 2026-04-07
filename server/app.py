"""
server/app.py
OpenEnv-required server entry point.

openenv validate checks:
  1. This file exists at server/app.py
  2. A main() function exists
  3. if __name__ == '__main__' block exists and calls main()

This file imports the FastAPI app from the root server.py
and starts uvicorn on port 7860 (HuggingFace Spaces default).
"""

import sys
import os

# Allow imports from project root (models, env, tasks, etc.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

# Import the FastAPI app defined in root server.py
from server import app  # noqa: F401  — re-exported for openenv discovery


def main() -> None:
    """
    Start the FakeNews OpenEnv server.
    Called by:
      - openenv validate (checks this function exists and is callable)
      - the 'fakenews-env-server' console script defined in pyproject.toml
      - docker CMD via: python -m server.app
    """
    uvicorn.run(
        "server:app",          # root-level server.py → app
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
