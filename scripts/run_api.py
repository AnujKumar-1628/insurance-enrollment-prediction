import argparse
import sys
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the prediction API.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the API server.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind the API server.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Restart the server when code changes. Use this during development.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
