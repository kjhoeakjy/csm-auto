"""m19. 수준 갱신 선택 규칙 빠른 시험 (재학습 제외)

각 판단 시점 T 에서
  1. 후보 수준 이동량을 모의 배포 시점 T' = T - 4분기 의 가용 자료로 적합
  2. T 에 새로 확정된 자료로 검증
       - 24개월 검증: T 에 확정된 최근 4개 분기 (기존 방식)
       - 조기 검증: T' 이후 새로 확정된 12개월 결과 4개 분기 (조기 기대 함수로 비교, 신규)
  3. 규칙에 따라 후보 선택, 통과 시 T 시점 자료로 다시 적합해 적용 (미통과 시 직전 적용 상태 유지)
보유 대출(24개월 결과 미확정 실행분) 오차는 평가에만 사용 (선택에 사용하지 않음)
실행: .venv/Scripts/python modelops/analysis/m19_select_lab.py --scores v2
"""
import argparse

import numpy as np
import pandas as pd

from config import DATA, RESULTS
from data import load_loans
from labels import HORIZON, LAG, REPORT_LAG
from m18_update_lab import LAST_LABELED, QUARTERS, lg, per_q_shift, sg, trend
from monitor import EarlyMap
from pipeline import asof_month, latest_labeled_q

CANDS = ["lag24_2q", "lag24_trend", "early12_2q", "early12_trend"]


def fit_shifts(df, asof, center_ord, m12):
    l24 = df[df["label_known_d"] <= asof]
    l12 = df[df["k12"] <= asof]
    last = lambda d, k: d[d["q"].isin(sorted(d["q"].unique())[-k:])]
    s24 = per_q_shift(last(l24, 4), "y", None)
    s12 = per_q_shift(last(l12, 4), "y12", m12)
    two = lambda s: float(np.mean([s[q] for q in sorted(s)[-2:]]))
    return {"lag24_2q": two(s24), "lag24_trend": trend(s24, max(s24).ordinal + 3),
            "early12_2q": two(s12), "early12_trend": trend(s12, center_ord)}


def oe(y, p):
    return y.sum() / p.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="v2")
    a = ap.parse_args()
    df = load_loans()
    sc = pd.read_parquet(DATA / f"scores_{a.scores}.parquet", columns=["id", "p_lgbm"])
    df = df.merge(sc, on="id", how="left").rename(columns={"p_lgbm": "p"})
    df["q"] = pd.PeriodIndex(df["issue_d"], freq="Q")
    df["k12"] = df["issue_d"] + pd.DateOffset(months=12 + LAG + REPORT_LAG)
    ref_end = "2012-12-31" if a.scores == "v1" else "2013-06-30"
    ref = df[(df["issue_d"] >= "2010-01-01") & (df["issue_d"] <= ref_end)]
    m12 = EarlyMap().fit(ref["p"], ref["y12"])
    rules = {"r4식 (24개월 검증)": ("lag",), "조기 검증": ("early",), "24개월+조기 검증 합산": ("lag", "early"),
             "조기 추세 매 분기": ("always",), "조기 추세, 이동 0.05 이상": ("delta", 0.05),
             "조기 추세, 이동 0.10 이상": ("delta", 0.10), "조기 추세, 조기 검증 개선 시": ("early_trend_gate",)}
    rows = []
    for rname, use in rules.items():
        cur = 0.0
        for t in QUARTERS:
            asof = asof_month(t)
            pseudo = asof - pd.DateOffset(months=12)
            q24 = latest_labeled_q(t, HORIZON)
            book = df[(df["q"] > q24) & (df["q"] <= min(t, LAST_LABELED))]
            center = (q24.ordinal + 1 + t.ordinal) / 2       # 운영 시점의 보유 대출 중심 (자료 끝과 무관)
            # 모의 배포: T' 기준 적합 (보유 대출 중심은 T' 기준으로 4분기 앞)
            sh_p = fit_shifts(df, pseudo, center - 4, m12)
            l24 = df[df["label_known_d"] <= asof]
            sel24 = l24[l24["q"].isin(sorted(l24["q"].unique())[-4:])]
            new12 = df[(df["k12"] <= asof) & (df["k12"] > pseudo)]
            def score(shift):
                s = 0.0
                if "lag" in use:
                    s += abs(oe(sel24["y"].values, sg(lg(sel24["p"].values) + shift)) - 1)
                if "early" in use:
                    s += abs(oe(new12["y12"].values, m12(sg(lg(new12["p"].values) + shift))) - 1)
                return s
            chosen = None
            if use[0] in ("always", "delta"):
                new = fit_shifts(df, asof, center, m12)["early12_trend"]
                if use[0] == "always" or abs(new - cur) >= use[1]:
                    chosen, cur = "early12_trend", new
            elif use[0] == "early_trend_gate":
                use_e = ("early",)
                s_new = abs(oe(new12["y12"].values, m12(sg(lg(new12["p"].values) + sh_p["early12_trend"]))) - 1)
                s_cur = abs(oe(new12["y12"].values, m12(sg(lg(new12["p"].values) + cur))) - 1)
                if s_new < s_cur:
                    chosen, cur = "early12_trend", fit_shifts(df, asof, center, m12)["early12_trend"]
            else:
                base = score(cur)
                best = min(CANDS, key=lambda k: score(sh_p[k]))
                if score(sh_p[best]) < base * 0.9:          # 현행 대비 오차 10% 이상 감소 시 갱신
                    chosen = best
                    cur = fit_shifts(df, asof, center, m12)[best]
            act = book["y"].mean()
            pred = sg(lg(book["p"].values) + cur).mean()
            rows.append({"rule": rname, "quarter": str(t), "chosen": chosen, "shift": cur,
                         "actual": act, "pred": pred, "err": abs(act / pred - 1)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"m19_select_lab_{a.scores}.csv", index=False, encoding="utf-8-sig")
    print((out.pivot(index="quarter", columns="rule", values="err") * 100).round(1).to_string())
    print(out.pivot(index="quarter", columns="rule", values="chosen").to_string())
    print("평균 오차 (%)\n", (out.groupby("rule")["err"].mean() * 100).round(1).to_string())
    print("갱신 횟수\n", out.groupby("rule")["chosen"].count().to_string())


if __name__ == "__main__":
    main()
