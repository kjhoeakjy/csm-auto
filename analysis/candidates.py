"""개선 후보 엔진: 유지 / 재보정 5종 / 재학습 3종을 같은 기준으로 비교

입력은 시점 T 에 정답이 확정된 코호트뿐 (호출부 asof_view 에서 잘라서 넘기고 여기서 다시 확인).
r2 모의 배포 검증:
  선택용(SEL) = T 에 24개월 정답이 확정된 최근 4분기
  모의 배포 시점 T' = T - 4분기. 후보는 T' 에 가용했던 정답만으로 적합
    재보정: T' 기준 24개월 확정 최근 2분기 / 조기 반영: T' 기준 12개월 확정 최근 2분기
    재학습: T' 기준 24개월 확정 최근 8분기
  배포 후보 = 같은 방식으로 T 시점 가용 데이터 전체에서 다시 적합. 미래 코호트 평가는 posthoc.py 에서만.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import features as F
from config import SEED
from metrics import evaluate
from models import lgb_model, logit_pipeline
from rules import R

EPS = 1e-6
COMPLEXITY = {"maintain": 0, "recal_intercept": 1, "recal_early": 2, "recal_platt": 3,
              "recal_beta": 4, "recal_grade": 5, "recal_isotonic": 6, "retrain_logit": 7,
              "retrain_lgbm": 8, "retrain_lgbm_ext": 9}
LABEL = {"maintain": "현행 유지", "recal_intercept": "전역 재보정(수준)",
         "recal_early": "조기 결과 반영 재보정", "recal_platt": "전역 재보정(Platt)",
         "recal_beta": "전역 재보정(beta)", "recal_grade": "등급별 재보정",
         "recal_isotonic": "전역 재보정(isotonic)", "retrain_logit": "재학습(로지스틱)",
         "retrain_lgbm": "재학습(LightGBM)", "retrain_lgbm_ext": "재학습(LightGBM+확장 변수)"}
KIND = {k: ("keep" if k == "maintain" else "recal" if k.startswith("recal") else "retrain")
        for k in COMPLEXITY}


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------- 재보정 ----------------
class Intercept:
    def fit(self, p, y, g=None):
        target = np.mean(y)
        self.a_ = brentq(lambda a: _sigmoid(_logit(p) + a).mean() - target, -5, 5)
        return self

    def predict(self, p, g=None):
        return _sigmoid(_logit(p) + self.a_)


class Platt:
    def fit(self, p, y, g=None):
        self.m_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p, g=None):
        return self.m_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class Beta:
    """beta calibration (Kull et al. 2017): logit q = a ln p - b ln(1-p) + c"""
    def _x(self, p):
        p = np.clip(p, EPS, 1 - EPS)
        return np.c_[np.log(p), -np.log(1 - p)]

    def fit(self, p, y, g=None):
        self.m_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._x(p), y)
        return self

    def predict(self, p, g=None):
        return self.m_.predict_proba(self._x(p))[:, 1]


class Isotonic:
    def fit(self, p, y, g=None):
        self.m_ = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p, y)
        return self

    def predict(self, p, g=None):
        return self.m_.predict(p)


class GradeIntercept:
    """등급별 절편 이동. 상각 50건 미만 등급은 전역 이동으로 대체"""
    def fit(self, p, y, g):
        self.glob_ = Intercept().fit(p, y).a_
        self.a_ = {}
        for k in np.unique(g):
            m = g == k
            if y[m].sum() >= 50:
                self.a_[k] = Intercept().fit(p[m], y[m]).a_
        return self

    def predict(self, p, g):
        a = np.array([self.a_.get(k, self.glob_) for k in g])
        return _sigmoid(_logit(p) + a)


RECAL = {"recal_intercept": Intercept, "recal_platt": Platt, "recal_beta": Beta,
         "recal_grade": GradeIntercept, "recal_isotonic": Isotonic}


# ---------------- 재학습 ----------------
class Retrain:
    def __init__(self, kind, champion_meta):
        self.kind, self.meta = kind, champion_meta

    def fit(self, d):
        ext = self.kind == "retrain_lgbm_ext"
        self.cols_ = F.num_cols(ext) + F.CAT_COLS
        X = F.build(d, extended=ext)[self.cols_]
        if self.kind == "retrain_logit":
            self.m_ = logit_pipeline(F.num_cols(), F.CAT_COLS, self.meta["logit"]["C"]).fit(X, d["y"])
        else:   # 챔피언과 같은 구조·반복 수 (선택용 구간으로 다시 튜닝하지 않음)
            self.m_ = lgb_model(n_estimators=self.meta["lgbm"]["n_estimators"],
                                **self.meta["lgbm"]["params"]).fit(X, d["y"])
        return self

    def predict(self, d):
        ext = self.kind == "retrain_lgbm_ext"
        return self.m_.predict_proba(F.build(d, extended=ext)[self.cols_])[:, 1]


# ---------------- 평가 ----------------
def _grade_rel(y, p, g):
    out = {}
    for k in np.unique(g):
        m = g == k
        if m.sum() >= 1000 and y[m].sum() >= 50:
            out[k] = y[m].sum() / p[m].sum() - 1
    return out


def _ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_delta(y, p_base, p_new, b, seed=SEED):
    """대출 단위 재표본으로 log loss 차이(new - base)의 95% 구간"""
    d = _ll(y, p_new) - _ll(y, p_base)
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(b)])
    return float(d.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


class EarlyIntercept:
    """조기 결과 반영 재보정: 12개월 실제 상각과 12개월 기대(early_map)가 같아지도록 24개월 logit 이동"""
    def __init__(self, early_map):
        self.emap = early_map

    def fit(self, p, y12, g=None):
        target = np.sum(y12)
        self.a_ = brentq(lambda a: self.emap(_sigmoid(_logit(p) + a)).sum() - target, -5, 5)
        return self

    def predict(self, p, g=None):
        return _sigmoid(_logit(p) + self.a_)


def _last_q(d, k):
    qs = sorted(d["q"].unique())
    return qs[-k:]


def _fit_all(lab24, lab12, meta, emap, with_retrain):
    """한 시점(as-of)에서 가용한 정답만으로 모든 후보 적합"""
    fit_r_q = _last_q(lab24, R["recal_fit_quarters"])
    fit_t_q = _last_q(lab24, R["retrain_window_quarters"])
    fit_e_q = _last_q(lab12, R["early_fit_quarters"])
    fr = lab24[lab24["q"].isin(fit_r_q)]
    fe = lab12[lab12["q"].isin(fit_e_q)]
    fitted = {k: cls().fit(fr["p"].values, fr["y"].values, fr["grade"].values)
              for k, cls in RECAL.items()}
    fitted["recal_early"] = EarlyIntercept(emap).fit(fe["p"].values, fe["y12"].values)
    if with_retrain:
        ft = lab24[lab24["q"].isin(fit_t_q)]
        for k in ["retrain_logit", "retrain_lgbm", "retrain_lgbm_ext"]:
            d = ft if k != "retrain_lgbm_ext" else ft[ft["issue_d"] >= "2013-01-01"]
            if len(d) >= 50000:
                fitted[k] = Retrain(k, meta).fit(d)
    windows = {"recal": [str(q) for q in fit_r_q], "early": [str(q) for q in fit_e_q],
               "retrain": [str(q) for q in fit_t_q]}
    return fitted, windows


def run_candidates(lab24: pd.DataFrame, lab12: pd.DataFrame, asof: pd.Timestamp,
                   champion_meta: dict, early_map, with_retrain: bool = True,
                   problem: str = "calibration") -> dict:
    """lab24/lab12: 시점 T 에 24/12개월 정답이 확정된 대출. 반환: 비교표·선택·배포 후보"""
    assert (lab24["label_known_d"] <= asof).all(), "시점 T 이후 확정 24개월 정답이 들어옴"
    assert (lab12["label12_known_d"] <= asof).all(), "시점 T 이후 확정 12개월 정답이 들어옴"
    sel_q = _last_q(lab24, R["sel_quarters"])
    sel = lab24[lab24["q"].isin(sel_q)]
    # 모의 배포 시점 T' = T - 4분기: 그때 가용했던 정답만으로 적합
    pseudo = asof - pd.DateOffset(months=3 * R["pseudo_origin_quarters"])
    p24 = lab24[lab24["label_known_d"] <= pseudo]
    p12 = lab12[lab12["label12_known_d"] <= pseudo]
    assert not set(p24["q"].unique()) & set(sel_q), "선택용 코호트가 적합에 들어감"
    fitted, win_pseudo = _fit_all(p24, p12, champion_meta, early_map, with_retrain)

    ys, ps, gs = sel["y"].values, sel["p"].values, sel["grade"].values
    preds = {"maintain": ps}
    for k, m in fitted.items():
        preds[k] = apply_candidate(k, m, sel)
    base = evaluate(ys, ps)
    base_ll = base["logloss"]
    base_grp = _grade_rel(ys, ps, gs)
    rows = []
    for k, p in preds.items():
        e = evaluate(ys, p)
        grp = _grade_rel(ys, p, gs)
        worsen = max([abs(grp[g]) - abs(base_grp[g]) for g in grp if g in base_grp] or [0])
        dm, dlo, dhi = boot_delta(ys, ps, p, R["bootstrap_b"]) if k != "maintain" else (0, 0, 0)
        checks = {
            "gain": (base_ll - e["logloss"]) / base_ll >= R["min_rel_logloss_gain"],
            "gain_ci": dhi < 0,
            "calibrated": abs(e["o_e_ratio"] - 1) < abs(base["o_e_ratio"] - 1),
            "auc": e["auc"] >= base["auc"] - R["max_auc_drop"],
            "groups": worsen <= R["max_group_worsen"],
        }
        rows.append({"candidate": k, "label": LABEL[k], "kind": KIND[k],
                     "complexity": COMPLEXITY[k], "logloss": e["logloss"],
                     "rel_ll_gain": (base_ll - e["logloss"]) / base_ll,
                     "dll_lo": dlo, "dll_hi": dhi, "brier": e["brier"], "auc": e["auc"],
                     "o_e_ratio": e["o_e_ratio"], "gap_pp": e["gap_pp"],
                     "worst_grade_rel": max([abs(v) for v in grp.values()] or [np.nan]),
                     "group_worsen": worsen,
                     **{f"ok_{c}": v for c, v in checks.items()},
                     "passed": k != "maintain" and all(checks.values())})
    table = pd.DataFrame(rows).sort_values("complexity").reset_index(drop=True)
    passed = table[table["passed"]]
    chosen, residual, note = None, None, "통과 후보 없음: 변경 보류"

    def simplest_best(d):
        if d.empty:
            return None
        near = d[d["logloss"] <= d["logloss"].min() * (1 + R["simpler_tolerance"])]
        return near.sort_values("complexity").iloc[0]

    r = simplest_best(passed[passed["kind"] == "recal"])
    t = simplest_best(passed[passed["kind"] == "retrain"])
    if problem == "ranking":            # 순위 악화: 재보정으로는 회복 불가 -> 재학습 우선
        pick = t if t is not None else r
        note = "순위 악화: 재학습 우선" if t is not None else "순위 악화이나 재학습 미통과: 재보정"
    else:                               # 확률 수준 오차: 재보정 우선, 재학습은 추가 이득이 클 때만
        if r is not None and t is not None:
            big = (t["rel_ll_gain"] - r["rel_ll_gain"] >= R["retrain_extra_ll"]
                   and t["auc"] - r["auc"] >= R["retrain_extra_auc"])
            pick = t if big else r
            note = "재학습의 추가 개선이 커서 재학습 선택" if big else "확률 수준 오차: 재보정으로 충분"
        else:
            pick = r if r is not None else t
            note = ("확률 수준 오차: 재보정 선택" if r is not None
                    else "재보정 미통과, 재학습만 통과" if t is not None else note)
    if pick is not None:
        chosen = pick["candidate"]
        residual = bool(abs(pick["o_e_ratio"] - 1) > R["residual_oe_dev"])
    # 배포 후보: 같은 방식으로 T 시점 가용 데이터에서 다시 적합 (모든 후보, 사후 평가용 포함)
    deploy_all, win_deploy = _fit_all(lab24, lab12, champion_meta, early_map, with_retrain)
    return {"table": table, "chosen": chosen, "residual": residual, "selection_note": note,
            "problem": problem,
            "deploy": deploy_all.get(chosen) if chosen else None, "deploy_all": deploy_all,
            "sel_q": [str(q) for q in sel_q], "pseudo_asof": str(pseudo.date()),
            "windows_pseudo": win_pseudo, "windows_deploy": win_deploy, "base_sel": base}


def apply_candidate(name: str, model, d: pd.DataFrame) -> np.ndarray:
    """후보로 코호트 점수 계산 (선택·사후 평가·화면 공용)"""
    if name is None or name == "maintain":
        return d["p"].values
    if KIND[name] == "recal":
        return model.predict(d["p"].values, d["grade"].values)
    return model.predict(d)
