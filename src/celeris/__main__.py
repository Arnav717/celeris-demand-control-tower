"""One-command pipeline: `python -m celeris [stage ...]`.

Stages run in order: clean -> anomalies -> features -> forecast. With no argument all
available stages run.
"""
from __future__ import annotations

import argparse
import importlib
import time

STAGES = ["clean", "anomalies", "features", "forecast"]


def main() -> None:
    ap = argparse.ArgumentParser(prog="celeris")
    ap.add_argument("stages", nargs="*", choices=STAGES + ["all"], default=["all"])
    args = ap.parse_args()
    todo = STAGES if "all" in args.stages else [s for s in STAGES if s in args.stages]
    for stage in todo:
        try:
            mod = importlib.import_module(f"celeris.{'models' if stage == 'forecast' else stage}")
        except ModuleNotFoundError:
            print(f"[skip] {stage}: not implemented yet")
            continue
        if not hasattr(mod, "run"):
            print(f"[skip] {stage}: not implemented yet")
            continue
        t0 = time.time()
        print(f"[run ] {stage} ...", flush=True)
        mod.run()
        print(f"[done] {stage} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
