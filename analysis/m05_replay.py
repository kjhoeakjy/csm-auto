"""m05. 운영 재현: 2015Q4~2018Q4 13분기를 무인 배치로 실행 (pipeline.run_quarter)

실행: .venv/Scripts/python modelops/analysis/m05_replay.py [--quarters 2017Q2,2017Q3] [--force]
"""
import argparse

import pandas as pd

from config import RESULTS
from pipeline import Context, run_quarter

QUARTERS = [str(q) for q in pd.period_range("2015Q4", "2018Q4", freq="Q")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarters", default=",".join(QUARTERS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ctx = Context.load()
    rows = []
    for q in a.quarters.split(","):
        r = run_quarter(ctx, q, force=a.force)
        chosen = (r.get("candidates") or {}).get("chosen")
        rows.append({"quarter": q, "run": r["run_status"], "skipped": r.get("skipped", False),
                     "status": r["status"], "recommendation": r["recommendation"],
                     "reasons": ",".join(r.get("reasons", [])), "experiment": r.get("candidate_experiment"),
                     "chosen": chosen, "sec": r.get("seconds"), "error": r.get("error")})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "m05_replay.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
