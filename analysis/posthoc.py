"""사후 평가 (Champion-Challenger): 판단 시점 T 의 선택이 그 뒤 실제 결과에서도 나았는가

선택 로직(candidates.py)과 물리적으로 분리. 저장된 배포 후보(runs/<규칙>/<T>/candidates.joblib)만 읽음.
평가 대상 = T 시점에 24개월 결과가 미확정이던 보유 코호트 (마지막 확정 분기 다음 ~ min(T, 2016Q3))
  (a) T 시점에 12개월 조기 결과가 있던 코호트  (b) 조기 결과도 없던 코호트
판단 이후 새로 실행된 대출은 정답이 2016Q3 실행분까지만 있어 평가 범위 밖.

실행: .venv/Scripts/python modelops/analysis/posthoc.py --rule r2
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from candidates import LABEL, apply_candidate, boot_delta
from config import RESULTS
from metrics import evaluate
from pipeline import RUNS, Context, latest_labeled_q

LAST_LABELED = pd.Period("2016Q3", freq="Q")


def evaluate_decision(ctx, rule, q):
    t = pd.Period(q, freq="Q")
    d = RUNS / rule / q
    if not (d / "candidates.joblib").exists():
        return []
    art = joblib.load(d / "candidates.joblib")
    models = art.get("deploy_all") or {}
    if not models and art.get("deploy") is not None:       # r1 은 선택 후보만 저장
        models = {art["chosen"]: art["deploy"]}
    q24, q12 = latest_labeled_q(t, 24), latest_labeled_q(t, 12)
    end = min(t, LAST_LABELED)
    fut = ctx.df[(ctx.df["q"] > q24) & (ctx.df["q"] <= end)]
    rows = []
    for part, sub in [("전체", fut), ("(a) 조기 결과 있음", fut[fut["q"] <= q12]),
                      ("(b) 조기 결과 없음", fut[fut["q"] > q12])]:
        if len(sub) < 5000:
            continue
        y, p0 = sub["y"].values, sub["p"].values
        base = evaluate(y, p0)
        for name, m in [("maintain", None)] + list(models.items()):
            p = p0 if m is None else apply_candidate(name, m, sub)
            e = evaluate(y, p)
            dm, lo, hi = (0, 0, 0) if m is None else boot_delta(y, p0, p, 200)
            rows.append({"rule": rule, "asof_q": q, "chosen": art["chosen"],
                         "is_chosen": name == art["chosen"], "candidate": name,
                         "label": LABEL.get(name, name), "part": part,
                         "cohorts": f"{sub['q'].min()}~{sub['q'].max()}", "n": e["n"],
                         "actual": e["actual"], "pred": e["pred"], "o_e_ratio": e["o_e_ratio"],
                         "gap_pp": e["gap_pp"], "auc": e["auc"], "brier": e["brier"],
                         "logloss": e["logloss"],
                         "rel_ll_gain": (base["logloss"] - e["logloss"]) / base["logloss"],
                         "dll_lo": lo, "dll_hi": hi})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="r2")
    a = ap.parse_args()
    ctx = Context.load()
    rows = []
    for d in sorted((RUNS / a.rule).iterdir()):
        if d.is_dir():
            rows += evaluate_decision(ctx, a.rule, d.name)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"m06_posthoc_{a.rule}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 400)
    cols = ["asof_q", "chosen", "candidate", "part", "cohorts", "n", "actual", "pred",
            "o_e_ratio", "gap_pp", "auc", "rel_ll_gain", "dll_hi"]
    print(out[out["part"] == "전체"][cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
