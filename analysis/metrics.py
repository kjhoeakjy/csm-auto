"""성능·보정 지표 (모니터링·후보 비교·사후 평가가 같은 함수를 사용)"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_line(y, p):
    """logit(P(y=1)) = a + b * logit(p) 적합. 완전 보정이면 a=0, b=1"""
    import statsmodels.api as sm
    x = sm.add_constant(_logit(p))
    try:
        r = sm.GLM(y, x, family=sm.families.Binomial()).fit()
        return float(r.params[0]), float(r.params[1])
    except Exception:
        return np.nan, np.nan


def auc_se(auc, n_pos, n_neg):
    """Hanley-McNeil 표준오차"""
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    return np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
                    + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))


def evaluate(y, p) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    n, o, e = len(y), y.sum(), p.sum()
    out = {"n": int(n), "events": int(o), "actual": o / n, "pred": e / n,
           "gap_pp": (o - e) / n * 100, "o_e_ratio": o / e}
    # 예측이 맞다는 가정하에서 O-E 의 표준편차 -> 차이가 표본 변동으로 설명되는지
    sd = np.sqrt((p * (1 - p)).sum())
    out["calib_z"] = (o - e) / sd
    out["calib_p"] = float(2 * stats.norm.sf(abs(out["calib_z"])))
    # 실제 상각률 95% 신뢰구간 (Wilson)
    z = 1.96
    c = (o / n + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(o / n * (1 - o / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    out["actual_lo"], out["actual_hi"] = c - h, c + h
    if 0 < o < n:
        out["auc"] = roc_auc_score(y, p)
        se = auc_se(out["auc"], o, n - o)
        out["auc_lo"], out["auc_hi"] = out["auc"] - 1.96 * se, out["auc"] + 1.96 * se
        out["pr_auc"] = average_precision_score(y, p)
        out["brier"] = brier_score_loss(y, p)
        out["logloss"] = log_loss(y, p)
        out["calib_a"], out["calib_b"] = calibration_line(y, p)
    return out


def by_group(df, group, y="y", p="p") -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(group, observed=True):
        r = evaluate(g[y], g[p])
        r[group if isinstance(group, str) else "group"] = k
        rows.append(r)
    return pd.DataFrame(rows)


def psi(expected, actual, bins=10, edges=None):
    """Population Stability Index. 결측은 별도 구간. edges 를 주면 기준 구간 경계를 고정해 사용"""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if edges is None:
        ok = expected[~np.isnan(expected)]
        edges = np.unique(np.quantile(ok, np.linspace(0, 1, bins + 1))) if len(ok) else np.array([0.0])
        edges = np.r_[-np.inf, edges[1:-1], np.inf]

    def dist(x):
        m = np.isnan(x)
        h = np.histogram(x[~m], edges)[0]
        return np.r_[h, m.sum()] / max(len(x), 1)

    e, a = np.clip(dist(expected), 1e-4, None), np.clip(dist(actual), 1e-4, None)
    return float(((a - e) * np.log(a / e)).sum()), edges
