"""m12. 결과 확정 시점 가정 민감도: 상각 기록 지연 4 / 5 / 6개월일 때 분기별 판정·권고가 바뀌는가

지연 L 이 바뀌면 각 판단 분기에 '확정된' 코호트 범위가 달라짐 (24개월: 실행 후 24+L+1개월, 12개월: 12+L+1개월).
정답 정의(상환 중단 기준)는 L 과 무관. 후보 실험은 재보정 계열만 (재학습 제외, 속도) → 판정·문제 유형·재보정 선택 비교
산출: results/m12_label_timing.csv

실행: .venv/Scripts/python modelops/analysis/m12_label_timing.py
"""
import pandas as pd

import labels
import pipeline
from config import RESULTS
from monitor import EarlyMap, estimate_phi

QUARTERS = list(pd.period_range("2015Q4", "2018Q4", freq="Q"))


def main():
    ctx = pipeline.Context.load()
    base = ctx.df["issue_d"]
    rows = []
    for L in (4, 5, 6):
        pipeline.LAG = L                                  # latest_labeled_q 가 쓰는 지연
        ctx.df["label_known_d"] = base + pd.DateOffset(months=24 + L + labels.REPORT_LAG)
        ctx.df["label12_known_d"] = base + pd.DateOffset(months=12 + L + labels.REPORT_LAG)
        for t in QUARTERS:
            out, cand = pipeline.decide(ctx, t, with_retrain=False)
            rows.append({"lag": L, "quarter": str(t), "status": out["status"], "problem": out["problem_label"],
                         "recommendation": out["recommendation"],
                         "lag_cohort": out["lag"][0]["cohort_q"] if out["lag"] else None,
                         "lag_oe": out["lag"][0]["o_e_ratio"] if out["lag"] else None,
                         "early_cohort": out["early"][0]["cohort_q"] if out["early"] else None,
                         "chosen_recal": cand["chosen"] if cand is not None else None})
            print(L, t, rows[-1]["status"], rows[-1]["lag_cohort"], rows[-1]["chosen_recal"], flush=True)
    pipeline.LAG = labels.LAG
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "m12_label_timing.csv", index=False, encoding="utf-8-sig")
    w = d.pivot(index="quarter", columns="lag", values="status")
    same = (w[4] == w[5]) & (w[6] == w[5])
    print(w.to_string())
    print(f"판정 일치 분기: {int(same.sum())}/{len(w)}")


if __name__ == "__main__":
    main()
