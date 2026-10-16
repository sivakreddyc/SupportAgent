"""`support-agent` CLI.

Entry-point contract each owner must provide:
  support_agent.mocks.ticket_ui.app:app        FastAPI app            (Agent A)
  support_agent.mocks.user_service.app:app     FastAPI app            (Agent A)
  support_agent.pipeline.runtime:run           async def run(settings) (Agent B)
  support_agent/dashboard/app.py               Streamlit script        (Agent C)
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="support-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="run the agent runtime (scheduler, poller, queue, workers, ops API)")
    p_ui = sub.add_parser("ticket-ui", help="run the mock ticket UI")
    p_ui.add_argument("--port", type=int, default=8001)
    p_us = sub.add_parser("user-service", help="run the mock user info service")
    p_us.add_argument("--port", type=int, default=8002)
    p_db = sub.add_parser("dashboard", help="run the Streamlit ops dashboard")
    p_db.add_argument("--port", type=int, default=8501)
    args = parser.parse_args(argv)

    if args.cmd == "run":
        from support_agent.common.config import get_settings
        from support_agent.pipeline.runtime import run

        asyncio.run(run(get_settings()))
        return 0

    if args.cmd in ("ticket-ui", "user-service"):
        import uvicorn

        target = {
            "ticket-ui": "support_agent.mocks.ticket_ui.app:app",
            "user-service": "support_agent.mocks.user_service.app:app",
        }[args.cmd]
        uvicorn.run(target, host="127.0.0.1", port=args.port, log_level="warning")
        return 0

    if args.cmd == "dashboard":
        script = Path(__file__).parent / "dashboard" / "app.py"
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(script),
                "--server.port",
                str(args.port),
                "--server.headless",
                "true",
            ]
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
