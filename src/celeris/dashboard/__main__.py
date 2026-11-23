"""`python -m celeris.dashboard [--host 0.0.0.0] [--port 8501]` starts the control tower."""
from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(prog="celeris.dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args()
    print(f"Celeris control tower on http://{args.host}:{args.port}", flush=True)
    uvicorn.run("celeris.dashboard.server:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
