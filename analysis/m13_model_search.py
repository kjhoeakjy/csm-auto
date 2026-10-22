"""m13. 예측모형 성능 개선 탐색 (선택은 개발 구간 선택용 데이터로만, 시간외 결과는 보고용)

설정 1 (운영 모형): 적합 2010Q1~2012Q2 / 선택 2012Q3~Q4 / 시간외 2013Q1~2016Q3
설정 2 (재학습, 2017Q2 시점 가정): 적합 2013Q1~2014Q2 / 선택 2014Q3~Q4 / 시간외 2015Q1~2016Q3
비교: v1 입력 vs v2 입력(파생·지역·직업), 기본 구조 vs 무작위 탐색 튜닝, LightGBM vs 로지스틱 vs 평균 앙상블
산출: results/m13_model_search.csv, m13_best.json

실행: .venv/Scripts/python modelops/analysis/m13_model_search.py
"""
import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

import features as F
import features_v2 as F2
from config import RESULTS, SEED
from data import load_loans
from metrics import evaluate
from models import fit_logit

SETTINGS = {
    "운영 모형": (("2010-01-01", "2012-06-30"), ("2012-07-01", "2012-12-31"), ("2013-01-01", "2016-09-30"), False),
    "재학습(2017Q2)": (("2013-01-01", "2014-06-30"), ("2014-07-01", "2014-12-31"), ("2015-01-01", "2016-09-30"), True),
}
N_SEARCH = 16


def between(df, a, b):
    return (df["issue_d"] >= a) & (df["issue_d"] <= b)


def lgb(params, n=4000):
    base = dict(learning_rate=0.03, subsample=0.8, subsample_freq=1, random_state=SEED, n_jobs=8, verbose=-1)
    base.update(params)
    return LGBMClassifier(n_estimators=n, **base)


def fit_es(Xf, yf, Xs, ys, params, cat):
    m = lgb(params).fit(Xf, yf, eval_set=[(Xs, ys)], eval_metric="binary_logloss",
                        categorical_feature=cat, callbacks=[early_stopping(150, verbose=False)])
    return m, m.best_score_["valid_0"]["binary_logloss"], m.best_iteration_


def refit(Xf, yf, Xs, ys, params, n, cat):
    return lgb(params, n=int(n * 1.1)).fit(pd.concat([Xf, Xs]), np.r_[yf, ys], categorical_feature=cat)


def search_space(rng):
    return dict(num_leaves=int(rng.choice([15, 31, 63, 127])), min_child_samples=int(rng.choice([100, 200, 500, 1000])),
                colsample_bytree=float(rng.choice([0.5, 0.7, 0.9])), reg_lambda=float(rng.choice([1, 5, 20, 50])),
                min_split_gain=float(rng.choice([0.0, 0.01])))


def run_setting(df, name, fit_r, sel_r, oot_r, ext):
    fit, sel, oot = between(df, *fit_r), between(df, *sel_r), between(df, *oot_r)
    zenc = F2.Zip3Encoder().fit(df.loc[fit, "zip_code"], df.loc[fit, "y"])     # 적합 구간 정답만
    X1 = F.build(df, extended=ext)
    X2 = F2.build(df, zenc, extended=ext)
    c1 = F.num_cols(ext) + F.CAT_COLS
    c2 = c1 + F2.V2_NUM + ["state_code"]
    y = df["y"].values
    rows, preds = [], {}

    def record(label, p, extra=None):
        es, eo = evaluate(y[sel.values], p[sel.values]), evaluate(y[oot.values], p[oot.values])
        q = df.loc[oot, "q"].values
        qa = [evaluate(y[oot.values][q == k], p[oot.values][q == k])["auc"] for k in np.unique(q)]
        rows.append({"setting": name, "model": label, "sel_auc": es["auc"], "sel_ll": es["logloss"],
                     "oot_auc": eo["auc"], "oot_ll": eo["logloss"], "oot_brier": eo["brier"],
                     "oot_oe": eo["o_e_ratio"], "oot_auc_q_mean": float(np.mean(qa)), **(extra or {})})
        preds[label] = p
        print(name, label, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)

    base = dict(num_leaves=31, min_child_samples=200, colsample_bytree=0.7, reg_lambda=5.0)
    for lab, X, cols, cat in [("LightGBM v1 입력", X1, c1, "auto"), ("LightGBM v2 입력", X2, c2, ["state_code"])]:
        m, ll, it = fit_es(X.loc[fit, cols], y[fit.values], X.loc[sel, cols], y[sel.values], base, cat)
        # 선택용 성능은 조기 종료 모형(선택용 미사용 적합)으로 기록, 시간외는 적합+선택 재적합으로
        p_sel = m.predict_proba(X[cols])[:, 1]
        mf = refit(X.loc[fit, cols], y[fit.values], X.loc[sel, cols], y[sel.values], base, it, cat)
        p = mf.predict_proba(X[cols])[:, 1]
        p = np.where(sel.values, p_sel, p)
        record(lab, p, {"iters": it})

    rng = np.random.default_rng(SEED)
    best = None
    for i in range(N_SEARCH):
        prm = search_space(rng)
        m, ll, it = fit_es(X2.loc[fit, c2], y[fit.values], X2.loc[sel, c2], y[sel.values], prm, ["state_code"])
        if best is None or ll < best[0]:
            best = (ll, it, prm, m)
    ll, it, prm, m = best
    p_sel = m.predict_proba(X2[c2])[:, 1]
    mf = refit(X2.loc[fit, c2], y[fit.values], X2.loc[sel, c2], y[sel.values], prm, it, ["state_code"])
    p_lgb = np.where(sel.values, p_sel, mf.predict_proba(X2[c2])[:, 1])
    record("LightGBM v2 입력 + 튜닝", p_lgb, {"iters": it, "params": json.dumps(prm)})

    num2 = F.num_cols(ext) + [c for c in F2.V2_NUM]
    lg, info = fit_logit(X2.loc[fit, num2 + F.CAT_COLS], y[fit.values], X2.loc[sel, num2 + F.CAT_COLS],
                         y[sel.values], num2, F.CAT_COLS)
    p_lr = lg.predict_proba(X2[num2 + F.CAT_COLS])[:, 1]
    record("로지스틱 v2 입력", p_lr, {"C": info["C"]})
    for w in (0.7, 0.85):
        record(f"앙상블 (LightGBM {w:.2f} + 로지스틱)", w * p_lgb + (1 - w) * p_lr)
    return rows, prm, it


def main():
    t0 = time.time()
    df = F2.attach_text(load_loans(end="2016-12-31"))
    df["q"] = df["issue_d"].dt.to_period("Q").astype(str)
    out, best = [], {}
    for name, (fr, sr, orr, ext) in SETTINGS.items():
        rows, prm, it = run_setting(df, name, fr, sr, orr, ext)
        out += rows
        best[name] = {"params": prm, "iters": it}
    res = pd.DataFrame(out)
    res.to_csv(RESULTS / "m13_model_search.csv", index=False, encoding="utf-8-sig")
    (RESULTS / "m13_best.json").write_text(json.dumps({"best": best, "seconds": round(time.time() - t0)},
                                                      ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(res.drop(columns=[c for c in ["params"] if c in res]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
