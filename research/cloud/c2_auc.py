"""클라우드 실험 2: 기존(운영) 모형 구분력 개선 — 첫 판단 시점(2015Q4 말)에 확정된 자료만 사용
적합 가능 코호트: 정답 확정일 <= 2015-12-31  -> 2013Q2 실행분까지
평가: 2013Q3~2016Q3 실행 코호트(보유 대출 구간) 시간외 AUC·logloss·실제/예측
변형
  B0 현행: 2010Q1~2012Q4, 핵심 변수, 128회
  B1 창 확대: 2010Q1~2013Q2, 핵심, 조기 종료 튜닝(창 마지막 2분기)
  B2 창 확대 + 확장 변수(2012-08 이후만 존재, 결측은 LightGBM 처리)
  B3 B2 + 적대적 검증 가중: 적합창 vs 가장 최근 신규 대출(2015Q3~Q4, 정답 불필요) 판별 모형 -> 밀도비 가중
  B4 B2 + 시간 가중(최근 분기 반감기 4분기)
  B5 B3 + B4
  B6 B2 창을 2012Q3~2013Q2 (확장 변수 완비 구간만)
"""
import time
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier, early_stopping
import features as F
from config import RESULTS, SEED
from data import load_loans
from labels import HORIZON
from metrics import evaluate
from models import lgb_model

OUT = RESULTS / "c2_auc.csv"
ASOF = pd.Timestamp("2015-12-31")
t0 = time.time()
df = load_loans()
df["q"] = pd.PeriodIndex(df["issue_d"], freq="Q")
Xc = F.build(df); Xe = F.build(df, extended=True)
lab = df["label_known_d"] <= ASOF
ev = (df["q"] >= "2013Q3") & (df["q"] <= "2016Q3")
new = (df["q"] >= "2015Q3") & (df["q"] <= "2015Q4")          # 판단 시점의 최근 신규 대출 (입력만 사용)
print("rows", len(df), "labeled", lab.sum(), "eval", ev.sum(), flush=True)

def window(a, b): return lab & (df["q"] >= a) & (df["q"] <= b)

def fit(X, mask, w=None, tune=True, params=dict(num_leaves=31, min_child_samples=200), n_fixed=128):
    y = df.loc[mask, "y"].values; Xm = X[mask]
    if not tune:
        m = lgb_model(n_estimators=n_fixed, n_jobs=2, **params).fit(Xm, y, sample_weight=w); return m, n_fixed
    qs = df.loc[mask, "q"]; last2 = sorted(qs.unique())[-2:]; va = qs.isin(last2).values
    m = lgb_model(n_estimators=3000, n_jobs=2, **params).fit(Xm[~va], y[~va], sample_weight=None if w is None else w[~va],
                                                              eval_set=[(Xm[va], y[va])], eval_metric="binary_logloss",
                                                              callbacks=[early_stopping(100, verbose=False)])
    n = int(m.best_iteration_ * 1.1)
    return lgb_model(n_estimators=n, n_jobs=2, **params).fit(Xm, y, sample_weight=w), n

def av_weights(X, mask):
    """적대적 검증: 적합창(0) vs 최근 신규(1) 판별 -> 적합창 행 가중 = p/(1-p) (0.2~5 로 제한, 평균 1)"""
    idx = np.r_[np.where(mask)[0], np.where(new.values)[0]]
    z = np.r_[np.zeros(mask.sum()), np.ones(new.sum())]
    clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=SEED, n_jobs=2, verbose=-1)
    rng = np.random.default_rng(SEED); fold = rng.integers(0, 3, len(idx)); p = np.zeros(len(idx))
    for f in range(3):
        tr = fold != f
        clf.fit(X.iloc[idx[tr]], z[tr]); p[~tr] = clf.predict_proba(X.iloc[idx[~tr]])[:, 1]
    auc_av = evaluate(z, p)["auc"]
    pw = p[: mask.sum()]
    w = np.clip(pw / (1 - pw) * (mask.sum() / new.sum()), 0.2, 5.0); w = w / w.mean()
    return w, auc_av

def time_w(mask, half=4):
    qs = df.loc[mask, "q"]; order = {k: i for i, k in enumerate(sorted(qs.unique()))}
    idx = qs.map(order).values; return 0.5 ** ((idx.max() - idx) / half)

def run(name, X, mask, w=None, tune=True, note=""):
    m, n = fit(X, mask, w, tune)
    p = m.predict_proba(X[ev])[:, 1]; e = evaluate(df.loc[ev, "y"].values, p)
    byq = [evaluate(df.loc[ev & (df.q == q), "y"].values, m.predict_proba(X[ev & (df.q == q)])[:, 1])["auc"]
           for q in sorted(df.loc[ev, "q"].unique())]
    row = {"variant": name, "n_fit": int(mask.sum()), "n_iter": n, "auc": e["auc"], "logloss": e["logloss"],
           "oe": e["o_e_ratio"], "auc_q_min": min(byq), "auc_q_last": byq[-1], "note": note, "sec": round(time.time() - t0)}
    print(row, flush=True); rows.append(row); pd.DataFrame(rows).to_csv(OUT, index=False)

rows = []
w0 = window("2010Q1", "2012Q4"); w1 = window("2010Q1", "2013Q2"); w6 = window("2012Q3", "2013Q2")
run("B0 현행 2010~12 핵심 128회", Xc, w0, tune=False)
run("B1 2010~13Q2 핵심 튜닝", Xc, w1)
run("B2 2010~13Q2 핵심+확장 튜닝", Xe, w1)
wav, auc_av = av_weights(Xe, w1.values)
run("B3 B2 + 적대적 검증 가중", Xe, w1, w=wav, note=f"판별 AUC {auc_av:.3f}")
wt = time_w(w1)
run("B4 B2 + 시간 가중(반감 4분기)", Xe, w1, w=wt)
run("B5 B2 + 적대적 + 시간 가중", Xe, w1, w=wav * wt)
run("B6 2012Q3~13Q2 핵심+확장 튜닝", Xe, w6)
print(pd.DataFrame(rows)[["variant", "auc", "logloss", "oe", "auc_q_last"]].to_string())
