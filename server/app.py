"""OpenEnv serve entry point. Re-exports the FastAPI app."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_server import app  # noqa: E402, F401

__all__ = ["app"]


def main() -> None:
    """Entry point for ``python -m server.app`` and ``[project.scripts]``."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()
