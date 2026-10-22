"""선행 점검(신규 실행 대출, 정답 불필요)과 후행 검증(정답 확정 코호트)

두 신호는 섞지 않음. 선행 점검은 '입력·점수가 달라졌는가', 후행 검증은 '예측이 실제와 맞았는가'.
모든 함수는 시점 T 에 쓸 수 있는 데이터만 받도록 호출부(pipeline)에서 잘라서 넘김.
"""
import numpy as np
import pandas as pd

import features as F
from metrics import evaluate, psi

MIN_N = 1000          # 집단별 판단 최소 표본
MIN_EVENTS = 50
EXPECTED_LEVELS = {
    "home_ownership": {"MORTGAGE", "RENT", "OWN", "OTHER", "NONE", "ANY"},
    "verification_status": {"Not Verified", "Verified", "Source Verified"},
    "application_type": {"Individual", "Joint App"},
    "term_m": {36.0, 60.0},
}
MIX_COLS = ["grade", "term_m", "purpose", "application_type", "verification_status"]


def _cat_psi(ref, cur):
    levels = sorted(set(ref.dropna().unique()) | set(cur.dropna().unique()), key=str)
    e = ref.value_counts(normalize=True).reindex(levels, fill_value=0).clip(lower=1e-4)
    a = cur.value_counts(normalize=True).reindex(levels, fill_value=0).clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum()), (a - e).to_dict()


def schema_check(cur: pd.DataFrame) -> dict:
    """필수 열 존재·형식·새 범주 확인. 오류가 있으면 판정 보류 대상"""
    errors, notes = [], []
    need = F.RAW_COLS
    missing_cols = [c for c in need if c not in cur.columns]
    if missing_cols:
        errors.append(f"필수 열 없음: {missing_cols[:5]}")
    for c in [c for c in F.CORE_NUM if c in cur.columns]:
        if not pd.api.types.is_numeric_dtype(cur[c]):
            errors.append(f"수치형이 아님: {c}")
    for c, allowed in EXPECTED_LEVELS.items():
        if c in cur.columns:
            new = set(cur[c].dropna().unique()) - allowed
            if new:
                notes.append(f"{c} 새 범주 {sorted(map(str, new))}")
    return {"ok": not errors, "errors": errors, "notes": notes}


def lead_check(cur: pd.DataFrame, ref: pd.DataFrame, recent: pd.DataFrame | None,
               score="p") -> dict:
    """cur: 이번 분기 신규 실행 대출 / ref: 학습 코호트 / recent: 직전 4분기 신규 대출"""
    out = {"n": int(len(cur)), "schema": schema_check(cur)}
    if not out["schema"]["ok"]:
        return out
    Xr, Xc = F.build(ref), F.build(cur)
    Xp = F.build(recent) if recent is not None and len(recent) else None
    rows = []
    for c in F.CORE_NUM:
        p_ref, edges = psi(Xr[c], Xc[c])
        r = {"feature": c, "psi_ref": p_ref,
             "miss_cur": float(Xc[c].isna().mean()), "miss_ref": float(Xr[c].isna().mean()),
             "mean_ref": float(Xr[c].mean()), "mean_cur": float(Xc[c].mean())}
        if Xp is not None:
            r["psi_recent"] = psi(Xp[c], Xc[c])[0]
            r["miss_recent"] = float(Xp[c].isna().mean())
        rows.append(r)
    feat = pd.DataFrame(rows)
    feat["miss_delta_ref_pp"] = (feat["miss_cur"] - feat["miss_ref"]) * 100
    if Xp is not None:
        feat["miss_delta_recent_pp"] = (feat["miss_cur"] - feat["miss_recent"]) * 100
    out["features"] = feat
    out["score"] = {"psi_ref": psi(ref[score], cur[score])[0],
                    "mean_ref": float(ref[score].mean()), "mean_cur": float(cur[score].mean())}
    if recent is not None and len(recent):
        out["score"]["psi_recent"] = psi(recent[score], cur[score])[0]
        out["score"]["mean_recent"] = float(recent[score].mean())
        out["volume_vs_recent"] = len(cur) / (len(recent) / 4)
    mix = {}
    for c in MIX_COLS:
        p_ref, d_ref = _cat_psi(ref[c], cur[c])
        mix[c] = {"psi_ref": p_ref, "share_delta_ref": d_ref}
        if recent is not None and len(recent):
            mix[c]["psi_recent"] = _cat_psi(recent[c], cur[c])[0]
    out["mix"] = mix
    return out


def lag_check(cohort: pd.DataFrame, phi: float = 1.0, score="p") -> dict:
    """정답 확정 코호트의 예측 대비 실제. phi = 분기 단위 과산포 (개발 구간에서 추정)"""
    r = evaluate(cohort["y"], cohort[score])
    r["z_adj"] = r["calib_z"] / np.sqrt(phi)
    r["rel_err"] = r["o_e_ratio"] - 1
    groups = {}
    for g in ["grade", "term_m", "purpose"]:
        rows = []
        for k, d in cohort.groupby(g, observed=True):
            if len(d) < MIN_N or d["y"].sum() < MIN_EVENTS:
                rows.append({g: k, "n": len(d), "status": "표본 부족"})
                continue
            e = evaluate(d["y"], d[score])
            rows.append({g: k, "n": e["n"], "actual": e["actual"], "pred": e["pred"],
                         "gap_pp": e["gap_pp"], "o_e_ratio": e["o_e_ratio"],
                         "z_adj": e["calib_z"] / np.sqrt(phi), "auc": e.get("auc"),
                         "status": "ok"})
        groups[g] = pd.DataFrame(rows)
    r["groups"] = groups
    return r


class EarlyMap:
    """24개월 예측 확률 p -> 12개월 조기 상각 기대 확률. 개발 구간에서 단조 회귀로 적합"""
    def fit(self, p, y12):
        from sklearn.isotonic import IsotonicRegression
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4).fit(p, y12)
        return self

    def __call__(self, p):
        return self.iso_.predict(np.asarray(p, dtype=float))


def early_check(cohort: pd.DataFrame, early_map: EarlyMap, phi: float = 1.0,
                score="p", y="y12") -> dict:
    """조기 결과 점검: 실행 후 12개월 안 상환 중단 후 상각 (정답 확정 18개월)"""
    r = evaluate(cohort[y], early_map(cohort[score]))
    r["z_adj"] = r["calib_z"] / np.sqrt(phi)
    r["rel_err"] = r["o_e_ratio"] - 1
    return r


def estimate_phi(df: pd.DataFrame, score="p", by="issue_q", y="y") -> float:
    """분기별 O-E 의 과산포 = mean(z^2). 1 이면 이항 변동만 존재"""
    z = []
    for _, d in df.groupby(by):
        e = evaluate(d[y], d[score])
        z.append(e["calib_z"])
    return float(max(np.mean(np.square(z)), 1.0))


SEG_LABEL = {"grade": "등급", "term_m": "기간(개월)", "purpose": "목적", "fico_band": "FICO", "dti_band": "DTI"}


def segment_scan(cohort: pd.DataFrame, phi: float, score="p") -> list[dict]:
    """문제 집단 탐색: 집단의 실제/예측이 전체 수준 대비 얼마나 다른지 검정 (Benjamini-Hochberg)
    전체 수준 r = O/E 를 집단 기대에 곱한 뒤 차이를 봄 -> '전체 이동'이 아닌 '특정 집단 집중'만 남김"""
    from scipy import stats
    from rules import R
    d = cohort.assign(
        fico_band=pd.cut(cohort["fico"], [0, 680, 720, 900], right=False,
                         labels=["680 미만", "680~719", "720 이상"]).astype(str),
        dti_band=pd.cut(cohort["dti"], [-1, 15, 25, 1000], right=False,
                        labels=["15 미만", "15~25", "25 이상"]).astype(str))
    r_all = d["y"].sum() / d[score].sum()
    rows = []
    for g in SEG_LABEL:
        for k, s in d.groupby(g, observed=True):
            o, p = s["y"].sum(), s[score].values
            if len(s) < R["seg_min_n"] or o < R["seg_min_events"]:
                continue
            e = r_all * p.sum()
            z = (o - e) / np.sqrt(phi * (r_all * p * (1 - r_all * p)).sum())
            rows.append({"group": SEG_LABEL[g], "level": str(k), "n": int(len(s)),
                         "o_e_ratio": float(o / p.sum()), "rel_to_all": float(o / e - 1),
                         "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))})
    if not rows:
        return []
    t = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(t)
    t["bh"] = t["p"] <= (np.arange(1, m + 1) / m) * R["seg_fdr"]
    t["bh"] = t["bh"][::-1].cummax()[::-1]          # BH: 통과한 가장 큰 순위까지 모두 유의
    t["flag"] = t["bh"] & (t["rel_to_all"].abs() >= R["seg_rel"])
    return t.to_dict("records")
