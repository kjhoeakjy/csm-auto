"""예측모형: 로지스틱 기준선 + LightGBM. 튜닝은 시간순 선택용 구간에서만 수행"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import SEED


class Winsorize(BaseEstimator, TransformerMixin):
    """학습 구간 1%·99% 분위수로 극단값 자르기 (경계는 학습 구간에서만 정함)"""
    def __init__(self, q=0.01):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.q, axis=0)
        self.hi_ = np.nanquantile(X, 1 - self.q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def logit_pipeline(num, cat, C=1.0):
    pre = ColumnTransformer([
        ("num", Pipeline([("w", Winsorize()),
                          ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", "passthrough", cat),
    ])
    return Pipeline([("pre", pre),
                     ("lr", LogisticRegression(C=C, max_iter=3000))])


def fit_logit(X_fit, y_fit, X_sel, y_sel, num, cat, grid=(0.01, 0.1, 1.0)):
    """정규화 강도 C 를 선택용 구간 log loss 로 고른 뒤 적합+선택 구간 전체로 재적합"""
    scores = {}
    for C in grid:
        m = logit_pipeline(num, cat, C).fit(X_fit, y_fit)
        scores[C] = log_loss(y_sel, m.predict_proba(X_sel)[:, 1])
    best = min(scores, key=scores.get)
    final = logit_pipeline(num, cat, best).fit(pd.concat([X_fit, X_sel]),
                                                np.r_[y_fit, y_sel])
    return final, {"C": best, "sel_logloss": scores}


LGB_GRID = [dict(num_leaves=15, min_child_samples=400),
            dict(num_leaves=31, min_child_samples=200),
            dict(num_leaves=63, min_child_samples=400)]


def lgb_model(n_estimators=2000, **kw):
    base = dict(learning_rate=0.03, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.7, reg_lambda=5.0, random_state=SEED, n_jobs=8,
                verbose=-1)
    base.update(kw)
    return LGBMClassifier(n_estimators=n_estimators, **base)


def fit_lgbm(X_fit, y_fit, X_sel, y_sel, grid=LGB_GRID):
    """구조 후보별로 선택용 구간 조기 종료 -> 최적 구조·반복 수로 적합+선택 구간 재적합"""
    res = []
    for g in grid:
        m = lgb_model(**g).fit(X_fit, y_fit, eval_set=[(X_sel, y_sel)],
                               eval_metric="binary_logloss",
                               callbacks=[early_stopping(100, verbose=False)])
        res.append((m.best_score_["valid_0"]["binary_logloss"], m.best_iteration_, g))
    best = min(res, key=lambda r: r[0])
    n_iter = int(best[1] * 1.1)   # 데이터가 늘어난 만큼 소폭 증가
    final = lgb_model(n_estimators=n_iter, **best[2]).fit(
        pd.concat([X_fit, X_sel]), np.r_[y_fit, y_sel])
    return final, {"params": best[2], "best_iter": int(best[1]), "n_estimators": n_iter,
                   "sel_logloss": [(r[0], r[1], r[2]) for r in res]}
