"""클라우드 실험 1: 갱신 시 예측 수준(보유 대출 상각률)의 근접도 개선
13개 판단 시점(2015Q4~2018Q4)에서 각 시점에 가용한 정보만으로 수준 보정 방식을 적용하고
보유 대출(24개월 결과 미확정 코호트, 2016Q3 까지)의 실제÷예측 오차를 비교.
전략
  maintain        유지
  intercept2      최근 확정 24개월 코호트 2분기 절편 보정 (현행 recal_intercept)
  trend4          확정 4분기 절편 추세 연장 ahead=3 (현행 recal_trend)
  early12         12개월 조기 결과 2분기 절편 보정 (현행 recal_early, 개발구간 사상)
  nowcast_book_X  보유 코호트 자체의 관측 가능한 최장 조기 결과(h=6/9/12/18개월)로 24개월 수준 추정 후 절편 보정
                   X=ref: p->y_h 사상을 개발구간(2010~12)에서 적합 / X=rec: 최근 확정 24개월 코호트 4분기에서 적합
  nowcast_mix     nowcast_book_rec 와 intercept2 를 코호트 관측 개월 수로 가중 평균
"""
import json, sys, time
import numpy as np, pandas as pd
from scipy.optimize import brentq
from sklearn.isotonic import IsotonicRegression
import features as F
from config import RESULTS, SEED
from data import load_loans
from labels import HORIZON, LAG, REPORT_LAG, make_label
from models import lgb_model
from metrics import evaluate

OUT = RESULTS / "c1_level.csv"
HS = [6, 9, 12, 18]
BOOK_CAP = pd.Period("2016Q3", freq="Q")

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def sig(x): return 1 / (1 + np.exp(-x))

def latest_labeled_q(t, horizon):
    m = t.asfreq("M", "end") - (horizon + LAG + REPORT_LAG)
    q = m.asfreq("Q")
    return q if m == q.asfreq("M", "end") else q - 1

def asof_month(t):
    return t.asfreq("D", "end").to_timestamp().normalize()

class Iso:
    def fit(self, p, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=1e-4).fit(p, y); return self
    def __call__(self, p): return self.m.predict(np.asarray(p, float))

def shift_to_match(p, target_sum, mp=None):
    """emap(sig(logit p + a)) 합이 target 이 되는 a"""
    f = (lambda a: (mp(sig(logit(p) + a)) if mp else sig(logit(p) + a)).sum() - target_sum)
    try: return brentq(f, -6, 6)
    except ValueError: return 0.0

t0 = time.time()
df = load_loans()
df["q"] = pd.PeriodIndex(df["issue_d"], freq="Q")
for h in HS:
    df[f"y{h}"] = make_label(df, horizon=h)
    df[f"k{h}"] = df["issue_d"] + pd.DateOffset(months=h + LAG + REPORT_LAG)
X = F.build(df)
ref = (df["issue_d"] >= "2010-01-01") & (df["issue_d"] <= "2012-12-31")
m = lgb_model(n_estimators=128, num_leaves=31, min_child_samples=200, n_jobs=2).fit(X[ref], df.loc[ref, "y"])
df["p"] = m.predict_proba(X)[:, 1]
print("champion fit", ref.sum(), f"{time.time()-t0:.0f}s", flush=True)
maps_ref = {h: Iso().fit(df.loc[ref, "p"], df.loc[ref, f"y{h}"]) for h in HS}
keep = ["issue_d", "q", "p", "y", "label_known_d", "grade"] + [f"y{h}" for h in HS] + [f"k{h}" for h in HS]
df = df[keep].copy(); del X, m
import gc; gc.collect()

rows = []
for T in pd.period_range("2015Q4", "2018Q4", freq="Q"):
    asof = asof_month(T)
    lq = latest_labeled_q(T, HORIZON)
    lab24 = df[df["label_known_d"] <= asof]
    book = df[(df["q"] > lq) & (df["q"] <= min(T, BOOK_CAP))]
    act, n = book["y"].mean(), len(book)
    qs24 = sorted(lab24["q"].unique())
    fr2 = lab24[lab24["q"].isin(qs24[-2:])]
    fr4 = lab24[lab24["q"].isin(qs24[-4:])]
    strat = {}
    strat["maintain"] = 0.0
    strat["intercept2"] = shift_to_match(fr2["p"].values, fr2["y"].sum())
    # trend: 4분기 절편 추세
    a_q = np.array([shift_to_match(fr4[fr4.q == k]["p"].values, fr4[fr4.q == k]["y"].sum()) for k in qs24[-4:]])
    sl, ic = np.polyfit(np.arange(4), a_q, 1)
    strat["trend4"] = float(np.clip(ic + sl * (3 + 3), a_q.min() - 0.3, a_q.max() + 0.3))
    # early12 (현행): 12개월 정답 확정 코호트 최근 2분기, 개발구간 사상
    l12 = df[df["k12"] <= asof]; q12 = sorted(l12["q"].unique())[-2:]; fe = l12[l12["q"].isin(q12)]
    strat["early12"] = shift_to_match(fe["p"].values, fe["y12"].sum(), maps_ref[12])
    # nowcast_book: 보유 코호트별 관측 가능한 최장 h
    maps_rec = {h: Iso().fit(fr4["p"], fr4[f"y{h}"]) for h in HS}
    for tag, maps in [("ref", maps_ref), ("rec", maps_rec)]:
        tgt, pp, wts = 0.0, [], []
        for c, g in book.groupby("q"):
            hs = [h for h in HS if (g[f"k{h}"] <= asof).all()]
            if not hs:
                continue
            h = max(hs)
            # 코호트 c 의 24개월 수준 추정: p 를 a 만큼 이동해 h개월 사상이 관측과 맞도록
            a_c = shift_to_match(g["p"].values, g[f"y{h}"].sum(), maps[h])
            tgt += sig(logit(g["p"].values) + a_c).sum(); pp.append(g["p"].values); wts.append((h, len(g)))
        if pp:
            pcat = np.concatenate(pp)
            strat[f"nowcast_book_{tag}"] = shift_to_match(pcat, tgt)
            cov = sum(n_ for _, n_ in wts) / n
        else:
            strat[f"nowcast_book_{tag}"] = strat["intercept2"]; cov = 0.0
    # rec2: 최근 확정 4분기에서 p->y_h, p->y24 두 사상을 적합. 보유 코호트의 관측 y_h 에 맞춘 이동을 p 에 가한 뒤 24개월 사상으로 수준 추정
    map24 = Iso().fit(fr4["p"], fr4["y"])
    for min_h, tag in [(6, "rec2"), (12, "rec2_h12")]:
        tgt, pp = 0.0, []
        for c, g in book.groupby("q"):
            hs = [h for h in HS if h >= min_h and (g[f"k{h}"] <= asof).all()]
            if not hs:
                continue
            h = max(hs)
            a_c = shift_to_match(g["p"].values, g[f"y{h}"].sum(), maps_rec[h])
            tgt += map24(sig(logit(g["p"].values) + a_c)).sum(); pp.append(g["p"].values)
        if pp:
            pcat = np.concatenate(pp); strat[f"nowcast_book_{tag}"] = shift_to_match(pcat, tgt)
        else:
            strat[f"nowcast_book_{tag}"] = strat["intercept2"]
    # mix: 관측 개월수 비율로 가중 (h/24)
    hs_w = [(h, n_) for h, n_ in wts] if pp else []
    w = (sum(h * n_ for h, n_ in hs_w) / (24 * max(n, 1))) if hs_w else 0.0
    strat["nowcast_mix"] = w * strat["nowcast_book_rec"] + (1 - w) * strat["intercept2"]
    for k, a in strat.items():
        pb = sig(logit(book["p"].values) + a)
        e = evaluate(book["y"].values, pb)
        rows.append({"T": str(T), "strategy": k, "a": round(a, 4), "book_n": n, "actual": act,
                     "pred": pb.mean(), "oe": e["o_e_ratio"], "abs_err": abs(e["o_e_ratio"] - 1),
                     "auc": e["auc"], "logloss": e["logloss"], "book_cover": cov, "w_mix": w})
    print(str(T), {k: round(v, 3) for k, v in strat.items()}, f"act={act:.4f}", f"{time.time()-t0:.0f}s", flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
summ = res.groupby("strategy").agg(mean_abs_err=("abs_err", "mean"), max_abs_err=("abs_err", "max"),
                                   mean_ll=("logloss", "mean")).sort_values("mean_abs_err")
print(summ.to_string())
summ.to_csv(RESULTS / "c1_level_summary.csv")
