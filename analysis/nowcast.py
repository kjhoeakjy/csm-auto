"""보유 대출 조기 결과 기반 수준 추정(nowcast) 재보정

24개월 정답이 아직 없는 보유 코호트의 관측 가능한 h개월(6, 9, 12, 18) 실제 상각 건수로
24개월 수준을 추정하고, 추정 합에 맞는 절편 하나를 24개월 점수에 가함.
  1. 최근 확정 24개월 코호트 4분기에서 p -> y_h 사상(h별)과 p -> y24 사상을 단조 회귀로 적합
  2. 보유 코호트별 관측 가능한 최장 h 를 택해 h개월 사상 합이 실제 h개월 상각 건수와 같도록 p 를 로짓 이동
  3. 이동한 p 를 24개월 사상에 넣어 코호트 24개월 수준을 추정, 전체 추정 합에 맞는 절편을 최종값으로 사용
근거: research/cloud/README.md (13개 판단 시점 평균 |실제÷예측-1| 0.086, 현행 재보정 0.17~0.19)
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.isotonic import IsotonicRegression

from labels import LAG, REPORT_LAG

HS = [6, 9, 12, 18]


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def add_early_labels(df: pd.DataFrame, make_label) -> pd.DataFrame:
    """y6, y9, y18 와 h개월 정답 확정월 k6..k18 부착 (y12 는 data.load_loans 가 부착)"""
    for h in HS:
        if f"y{h}" not in df.columns:
            df[f"y{h}"] = make_label(df, horizon=h)
        df[f"k{h}"] = df["issue_d"] + pd.DateOffset(months=h + LAG + REPORT_LAG)
    return df


def _shift_to_match(p, target_sum, mp=None):
    """mp(sigmoid(logit p + a)) 의 합이 target_sum 이 되는 a (해가 없으면 0)"""
    f = lambda a: (mp(_sigmoid(_logit(p) + a)) if mp is not None else _sigmoid(_logit(p) + a)).sum() - target_sum
    try:
        return float(brentq(f, -6, 6))
    except ValueError:
        return 0.0


class NowcastIntercept:
    """후보명 recal_nowcast. fit(ref24, book, asof): ref24 = 최근 확정 24개월 코호트(사상 적합용),
    book = 24개월 미확정 보유 코호트(y_h, k_h 열 필요), asof = 판단 시점"""
    def fit(self, ref24: pd.DataFrame, book: pd.DataFrame, asof: pd.Timestamp):
        maps = {h: IsotonicRegression(out_of_bounds="clip", y_min=1e-4).fit(ref24["p"], ref24[f"y{h}"]) for h in HS}
        map24 = IsotonicRegression(out_of_bounds="clip", y_min=1e-4).fit(ref24["p"], ref24["y"])
        target, ps, used = 0.0, [], []
        for q, g in book.groupby("q"):
            hs = [h for h in HS if (g[f"k{h}"] <= asof).all()]
            if not hs:
                continue
            h = max(hs)
            a_c = _shift_to_match(g["p"].values, g[f"y{h}"].sum(), maps[h].predict)
            target += map24.predict(_sigmoid(_logit(g["p"].values) + a_c)).sum()
            ps.append(g["p"].values)
            used.append({"q": str(q), "h": h, "n": int(len(g)), "shift": round(a_c, 4)})
        self.used_ = used
        if not ps:                       # 관측 가능한 보유 코호트 없음: 보정 없음
            self.a_ = 0.0
            return self
        self.a_ = _shift_to_match(np.concatenate(ps), target)
        return self

    def predict(self, p, g=None):
        return _sigmoid(_logit(p) + self.a_)
