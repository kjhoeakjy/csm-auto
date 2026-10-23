"""m18. 갱신 방식 빠른 시험 (재학습 제외, 분 단위): 각 판단 시점 T 에 가용한 정보만으로 수준 보정 후
그 시점 보유 대출(24개월 결과 미확정, 2016Q3 실행분까지)의 예측 오차 |실제/예측 - 1| 비교

후보 (모두 운영 모형 점수의 logit 수준 이동, T 에 확정된 정답만 사용)
  none        : 갱신 없음
  lag24_2q    : 24개월 확정 최근 2분기 수준 (기존 전역 재보정)
  lag24_trend : 24개월 확정 최근 4분기 수준 추세 연장 (기존 추세 반영 재보정)
  early12_2q  : 12개월 확정(실행 후 18개월) 최근 2분기 (기존 조기 결과 반영 재보정)
  early6_2q   : 6개월 확정(실행 후 12개월) 최근 2분기 (신규)
  early6_trend: 6개월 확정 최근 4분기 수준 추세를 보유 대출 중심 시점까지 연장 (신규)
  early_mix   : 6개월, 12개월 확정 최근 2분기 수준의 평균 (신규)
주의: 보유 대출 결과를 보고 방식을 고르는 탐색 단계 (사후 정보). 최종 선택 규칙은 pipeline 에서 모의 배포 검증
실행: .venv/Scripts/python modelops/analysis/m18_update_lab.py [--scores v1|v2]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from config import DATA, RESULTS
from data import load_loans
from labels import HORIZON, LAG, REPORT_LAG, make_label
from monitor import EarlyMap
from pipeline import asof_month, latest_labeled_q

EPS = 1e-6
QUARTERS = pd.period_range("2015Q4", "2018Q4", freq="Q")
LAST_LABELED = pd.Period("2016Q3", freq="Q")


def lg(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sg(x):
    return 1 / (1 + np.exp(-x))


def shift_to(p, target_sum, fmap=None):
    f = (lambda a: sg(lg(p) + a)) if fmap is None else (lambda a: fmap(sg(lg(p) + a)))
    return brentq(lambda a: f(a).sum() - target_sum, -5, 5)


def per_q_shift(d, ycol, fmap):
    out = {}
    for q, g in d.groupby("q"):
        out[q] = shift_to(g["p"].values, g[ycol].sum(), fmap)
    return out


def trend(sh, target_ord):
    qs = sorted(sh)
    x = np.array([q.ordinal for q in qs], dtype=float)
    a = np.array([sh[q] for q in qs])
    if len(qs) < 3:
        return a.mean()
    b, c = np.polyfit(x, a, 1)
    return float(np.clip(c + b * target_ord, a.min() - 0.3, a.max() + 0.3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="v1")
    a = ap.parse_args()
    df = load_loans()
    sc = pd.read_parquet(DATA / f"scores_{a.scores}.parquet", columns=["id", "p_lgbm"])
    df = df.merge(sc, on="id", how="left").rename(columns={"p_lgbm": "p"})
    df["q"] = pd.PeriodIndex(df["issue_d"], freq="Q")
    df["y6"] = make_label(df, horizon=6)
    k6 = df["issue_d"] + pd.DateOffset(months=6 + LAG + REPORT_LAG)
    k12 = df["issue_d"] + pd.DateOffset(months=12 + LAG + REPORT_LAG)
    ref_end = "2012-12-31" if a.scores == "v1" else "2013-06-30"
    ref = df[(df["issue_d"] >= "2010-01-01") & (df["issue_d"] <= ref_end)]
    m6, m12 = EarlyMap().fit(ref["p"], ref["y6"]), EarlyMap().fit(ref["p"], ref["y12"])
    rows = []
    for t in QUARTERS:
        asof = asof_month(t)
        q24 = latest_labeled_q(t, HORIZON)
        book = df[(df["q"] > q24) & (df["q"] <= min(t, LAST_LABELED))]
        center = np.mean([q.ordinal for q in book["q"].unique()])
        l24 = df[df["label_known_d"] <= asof]
        l12 = df[k12 <= asof]
        l6 = df[k6 <= asof]
        last = lambda d, k: d[d["q"].isin(sorted(d["q"].unique())[-k:])]
        s24 = per_q_shift(last(l24, 4), "y", None)
        s12 = per_q_shift(last(l12, 4), "y12", m12)
        s6 = per_q_shift(last(l6, 4), "y6", m6)
        two = lambda s: float(np.mean([s[q] for q in sorted(s)[-2:]]))
        shifts = {"none": 0.0, "lag24_2q": two(s24), "lag24_trend": trend(s24, max(s24).ordinal + 3),
                  "early12_2q": two(s12), "early6_2q": two(s6), "early6_trend": trend(s6, center),
                  "early12_trend": trend(s12, center), "early_mix": (two(s6) + two(s12)) / 2}
        act = book["y"].mean()
        for k, s in shifts.items():
            pred = sg(lg(book["p"].values) + s).mean()
            rows.append({"quarter": str(t), "method": k, "shift": s, "actual": act, "pred": pred,
                         "err": abs(act / pred - 1), "n": len(book)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"m18_update_lab_{a.scores}.csv", index=False, encoding="utf-8-sig")
    piv = out.pivot(index="quarter", columns="method", values="err")
    print((piv * 100).round(1).to_string())
    print("평균 오차 (%)\n", (piv.mean() * 100).round(1).sort_values().to_string())


if __name__ == "__main__":
    main()
