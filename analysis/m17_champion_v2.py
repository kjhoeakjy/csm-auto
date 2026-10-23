"""m17. 운영 모형 v2 탐색: 첫 판단 시점(2015Q4)에 결과가 확정된 자료 전체로 학습

- 2015Q4 판단 시점에 24개월 결과가 확정된 실행분: 2013Q2 까지 (실행 후 30개월)
- 비교 기준: 기존 운영 모형(2010~2012 학습). 평가는 모든 후보 공통 시간외 구간 2013Q3~2016Q3
- 시간순 분리: 적합 ~2012Q4, 선택(조기 종료·구조 선택) 2013Q1~Q2, 최종 = 적합+선택 재적합
- 후보: 학습 시작 연도(2008/2010), 입력(기본/신용조회 확장), 최근 가중(반감기), 구조 격자
산출: results/m17_champion_search.csv

실행: .venv/Scripts/python modelops/analysis/m17_champion_v2.py
"""
import time

import numpy as np
import pandas as pd
from lightgbm import early_stopping

import features as F
from config import RESULTS
from data import load_loans
from metrics import evaluate
from models import lgb_model

OOT = ("2013-07-01", "2016-09-30")
GRID = [dict(num_leaves=15, min_child_samples=400), dict(num_leaves=31, min_child_samples=200),
        dict(num_leaves=63, min_child_samples=400), dict(num_leaves=31, min_child_samples=800, reg_lambda=20.0),
        dict(num_leaves=127, min_child_samples=1000, colsample_bytree=0.5)]


def between(df, a, b):
    return (df["issue_d"] >= a) & (df["issue_d"] <= b)


def weights(issue_d, end, half_life_q):
    if not half_life_q:
        return None
    age_q = ((pd.Timestamp(end) - issue_d).dt.days / 91.3).clip(lower=0)
    return np.power(0.5, age_q / half_life_q).values


def fit_one(X, y, fit, sel, cols, w_fit, w_all):
    res = []
    for g in GRID:
        m = lgb_model(**g).fit(X.loc[fit, cols], y[fit], sample_weight=w_fit,
                               eval_set=[(X.loc[sel, cols], y[sel])], eval_metric="binary_logloss",
                               callbacks=[early_stopping(100, verbose=False)])
        res.append((m.best_score_["valid_0"]["binary_logloss"], m.best_iteration_, g))
    best = min(res, key=lambda r: r[0])
    final = lgb_model(n_estimators=int(best[1] * 1.1), **best[2]).fit(
        X.loc[fit | sel, cols], y[fit | sel], sample_weight=w_all)
    return final, best


def main():
    t0 = time.time()
    df = load_loans(start="2008-01-01")
    X = F.build(df, extended=True)
    y = df["y"].values
    oot = between(df, *OOT)
    rows = []
    cfgs = [("기존 v1 재현 (2010~2012, 기본 입력)", "2010-01-01", "2012-06-30", ("2012-07-01", "2012-12-31"), False, None),
            ("2010~2013Q2, 기본 입력", "2010-01-01", "2012-12-31", ("2013-01-01", "2013-06-30"), False, None),
            ("2008~2013Q2, 기본 입력", "2008-01-01", "2012-12-31", ("2013-01-01", "2013-06-30"), False, None),
            ("2010~2013Q2, 확장 입력", "2010-01-01", "2012-12-31", ("2013-01-01", "2013-06-30"), True, None),
            ("2010~2013Q2, 확장 입력, 최근 가중 8분기", "2010-01-01", "2012-12-31", ("2013-01-01", "2013-06-30"), True, 8),
            ("2010~2013Q2, 확장 입력, 최근 가중 4분기", "2010-01-01", "2012-12-31", ("2013-01-01", "2013-06-30"), True, 4)]
    scores = {}
    for name, a, b, (sa, sb), ext, hl in cfgs:
        fit, sel = between(df, a, b), between(df, sa, sb)
        cols = F.num_cols(ext) + F.CAT_COLS
        w_fit = weights(df.loc[fit, "issue_d"], sb, hl)
        w_all = weights(df.loc[fit | sel, "issue_d"], sb, hl)
        m, best = fit_one(X, y, fit, sel, cols, w_fit, w_all)
        p = m.predict_proba(X[cols])[:, 1]
        e = evaluate(y[oot], p[oot])
        q = df.loc[oot].assign(p=p[oot]).groupby("issue_q").apply(
            lambda g: evaluate(g["y"].values, g["p"].values)["auc"], include_groups=False)
        rows.append({"setting": name, "train_end": sb, "n_train": int((fit | sel).sum()), "params": str(best[2]),
                     "iters": int(best[1]), "sel_ll": best[0], "oot_auc": e["auc"], "oot_ll": e["logloss"],
                     "oot_brier": e["brier"], "oot_oe": e["o_e_ratio"], "oot_auc_q_mean": float(q.mean())})
        scores[name] = p
        print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "m17_champion_search.csv", index=False, encoding="utf-8-sig")
    print(out[["setting", "oot_auc", "oot_ll", "oot_oe", "oot_auc_q_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
