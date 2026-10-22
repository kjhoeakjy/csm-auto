"""분기 처리 함수: 스케줄러(정기 실행)와 과거 분기 재현이 같은 함수를 호출

run_quarter(T):
  as-of 자르기 -> 선행 점검 -> 조기 결과 점검 -> 후행 검증 -> 판정
  -> (조치 검토일 때만) 개선 후보 비교 -> 권고 + 근거 3개 -> 정량 근거 JSON 저장
실패(스키마 오류, 표본 부족, 예외)는 '판정 유보/실패'로 기록하고 보고서를 만들지 않음.
같은 입력(데이터·코드·모형·규칙 버전)으로 이미 성공한 분기는 다시 만들지 않음.
"""
import hashlib
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from candidates import KIND, LABEL, run_candidates
from config import DATA, MO, RESULTS
from data import load_loans
from labels import HORIZON, LAG, REPORT_LAG
from metrics import evaluate
from monitor import EarlyMap, early_check, estimate_phi, lag_check, lead_check, segment_scan
from rules import PROBLEM, R, RECO, RULE_VERSION, diagnose, judge

RUNS = MO / "runs"
RUNS.mkdir(exist_ok=True)
CODE_FILES = ["pipeline.py", "monitor.py", "rules.py", "candidates.py", "metrics.py",
              "features.py", "labels.py", "models.py", "data.py"]


def code_hash():
    h = hashlib.sha1()
    for f in CODE_FILES:
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:12]


def asof_month(t: pd.Period) -> pd.Timestamp:
    return t.asfreq("M", "end").to_timestamp()


def latest_labeled_q(t: pd.Period, horizon: int) -> pd.Period:
    m = t.asfreq("M", "end") - (horizon + LAG + REPORT_LAG)
    q = m.asfreq("Q")
    return q if m == q.asfreq("M", "end") else q - 1


@dataclass
class Context:
    df: pd.DataFrame
    ref: pd.DataFrame
    phi: float
    phi12: float
    emap: EarlyMap
    meta: dict
    data_id: str
    ref_auc: float = None

    @classmethod
    def load(cls, model_path=RESULTS / "models" / "champion_v1.joblib"):
        art = joblib.load(model_path)
        df = load_loans()
        sc = pd.read_parquet(DATA / "scores_v1.parquet", columns=["id", "p_lgbm"])
        df = df.merge(sc, on="id", how="left").rename(columns={"p_lgbm": "p"})
        df["q"] = pd.PeriodIndex(df["issue_d"], freq="Q")
        df["label12_known_d"] = df["issue_d"] + pd.DateOffset(months=12 + LAG + REPORT_LAG)
        ref = df[(df["issue_d"] >= "2010-01-01") & (df["issue_d"] <= "2012-12-31")]
        emap = EarlyMap().fit(ref["p"], ref["y12"])
        ref = ref.assign(p12=emap(ref["p"]))
        ingest = json.loads((RESULTS / "m01_ingest.json").read_text(encoding="utf-8"))
        # 기준 AUC: 운영 모형의 첫 시간외 코호트(2013Q1~Q2). 운영 재현 첫 분기(2015Q4)에 이미 확정
        oot = df[(df["issue_d"] >= "2013-01-01") & (df["issue_d"] <= "2013-06-30")]
        return cls(df=df, ref=ref, phi=estimate_phi(ref), phi12=estimate_phi(ref, "p12", y="y12"),
                   emap=emap, meta=art["meta"],
                   data_id=f"{ingest['source_sha1'][:10]}@{ingest['snapshot_last_pymnt_max']}",
                   ref_auc=evaluate(oot["y"], oot["p"])["auc"])


def asof_view(ctx: Context, t: pd.Period) -> dict:
    """시점 T 에 알 수 있었던 데이터만. 정답은 확정일 기준으로 다시 가림"""
    asof = asof_month(t)
    df = ctx.df
    lab24 = df[df["label_known_d"] <= asof]
    lab12 = df[df["label12_known_d"] <= asof]
    return {"asof": asof, "new": df[df["q"] == t],
            "recent": df[(df["q"] >= t - 4) & (df["q"] <= t - 1)],
            "lab24": lab24, "lab12": lab12}


def _lead_summary(r: dict) -> dict:
    if not r["schema"]["ok"]:
        return {"schema_ok": False, "schema_errors": r["schema"]["errors"], "n": r["n"]}
    f = r["features"]
    top_ref = f.nlargest(3, "psi_ref")
    top_rec = f.nlargest(3, "psi_recent")
    miss = f.loc[f["miss_delta_recent_pp"].abs().idxmax()]
    return {
        "schema_ok": True, "schema_notes": r["schema"]["notes"], "n": r["n"],
        "volume_vs_recent": r.get("volume_vs_recent"),
        "score_mean": r["score"]["mean_cur"], "score_mean_ref": r["score"]["mean_ref"],
        "score_psi_ref": r["score"]["psi_ref"], "score_psi_recent": r["score"]["psi_recent"],
        "n_feat_psi_recent_ge_0.25": int((f["psi_recent"] >= .25).sum()),
        "top_ref": [{"feature": a, "psi": b} for a, b in zip(top_ref.feature, top_ref.psi_ref)],
        "top_recent": [{"feature": a, "psi": b} for a, b in zip(top_rec.feature, top_rec.psi_recent)],
        "miss_max_delta_recent_pp": float(abs(miss["miss_delta_recent_pp"])),
        "miss_max_feature": miss["feature"],
        "grade_psi_ref": r["mix"]["grade"]["psi_ref"],
        "joint_share_delta_ref": r["mix"]["application_type"]["share_delta_ref"].get("Joint App", 0.0),
    }


def _outcome_summary(r: dict, cohort_q) -> dict:
    keys = ["n", "events", "actual", "pred", "gap_pp", "o_e_ratio", "calib_z", "z_adj",
            "rel_err", "auc", "auc_lo", "auc_hi", "brier", "logloss", "calib_a", "calib_b"]
    out = {k: r.get(k) for k in keys}
    out["cohort_q"] = str(cohort_q)
    if "groups" in r:
        g = r["groups"]["grade"]
        g = g[g["status"] == "ok"]
        out["grade"] = g[["grade", "n", "actual", "pred", "o_e_ratio"]].to_dict("records") if len(g) else []
    return out


def _fmt_pct(x):
    return f"{x * 100:.1f}%"


def evidence(lead, early, lag, status, cand) -> list[dict]:
    """근거 3개 (정량 근거에서 규칙으로 생성. 수치는 decision.json 과 같은 값)"""
    ev = []
    if lead.get("skipped"):
        ev.append({"id": "input", "title": "입력 분포·데이터 품질",
                   "text": f"{lead['skipped']} ({lead['n']:,}건): 선행 점검 유보"})
    elif lead.get("schema_ok"):
        t = lead["top_recent"][0]
        ev.append({"id": "input", "title": "입력 분포·데이터 품질",
                   "text": (f"신규 {lead['n']:,}건. 점수 분포 변화 PSI {lead['score_psi_ref']:.3f}(학습 대비), "
                            f"{lead['score_psi_recent']:.3f}(직전 4분기 대비). "
                            f"가장 크게 바뀐 변수 {t['feature']} (PSI {t['psi']:.2f})")})
    else:
        ev.append({"id": "input", "title": "입력 분포·데이터 품질",
                   "text": "스키마 오류로 점검 불가: " + "; ".join(lead.get("schema_errors", []))})
    l0 = lag[0]
    e0 = early[0] if early else None
    txt = (f"{l0['cohort_q']} 실행분(24개월 확정) 실제 {_fmt_pct(l0['actual'])} vs 예측 {_fmt_pct(l0['pred'])} "
           f"(실제/예측 {l0['o_e_ratio']:.2f})")
    if e0:
        txt += (f". 조기 점검 {e0['cohort_q']} 실행분(12개월) 실제 {_fmt_pct(e0['actual'])} vs 기대 "
                f"{_fmt_pct(e0['pred'])} (실제/기대 {e0['o_e_ratio']:.2f})")
    ev.append({"id": "outcome", "title": "예측과 실제의 차이", "text": txt})
    if cand is None:
        ev.append({"id": "candidate", "title": "개선 후보 검증",
                   "text": "조치 검토 기준 미달로 개선 후보 실험 생략"})
    else:
        t = cand["table"].set_index("candidate")
        m = t.loc["maintain"]
        if cand["chosen"]:
            c = t.loc[cand["chosen"]]
            txt = (f"1년 전 시점 데이터로 적합한 {LABEL[cand['chosen']]}을 그 뒤 확정된 "
                   f"{cand['sel_q'][0]}~{cand['sel_q'][-1]} 실행분에 적용: 실제/예측 {m['o_e_ratio']:.2f} → "
                   f"{c['o_e_ratio']:.2f}, log loss {c['rel_ll_gain'] * 100:.1f}% 개선, "
                   f"AUC {m['auc']:.3f} → {c['auc']:.3f}")
        else:
            txt = (f"후보 {len(t) - 1}개 모두 사전 기준 미달 (1년 전 시점 적합 → "
                   f"{cand['sel_q'][0]}~{cand['sel_q'][-1]} 실행분 평가). 변경 보류")
        ev.append({"id": "candidate", "title": "개선 후보 검증", "text": txt})
    return ev


def recommend(status, reasons, cand) -> str:
    if status == "판정 유보":
        return RECO["hold"]
    if status == "조치 검토":
        if cand is None or cand["chosen"] is None:
            return RECO["investigate"]
        r = RECO[KIND[cand["chosen"]]]
        return r + " (잔여 오차 있음·원인 조사 병행)" if cand.get("residual") else r
    if status == "주의":
        outcome = any(r.startswith(("lag24", "early12")) for r in reasons)
        return RECO["investigate"] if outcome else RECO["keep"]
    return RECO["keep"]


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Period, pd.Timestamp)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def input_key(ctx: Context, t) -> str:
    return f"{t}|{ctx.data_id}|{ctx.meta['model_id']}|{RULE_VERSION}|{code_hash()}"


def decide(ctx: Context, t, with_retrain=True, inject=None, ref_auc=None) -> tuple[dict, dict | None]:
    """한 분기 판단 (파일 기록 없음). 정책 비교 시뮬레이션(m11)도 이 함수를 그대로 사용"""
    t = pd.Period(t, freq="Q")
    v = asof_view(ctx, t)
    if inject is not None:
        v = inject(v)
    out = {"asof": str(v["asof"].date())}
    if len(v["new"]) < R["min_new_n"]:           # 신규 데이터 미도착·부족: 선행 점검만 유보
        lead = {"schema_ok": True, "skipped": "신규 실행 데이터 부족", "n": int(len(v["new"]))}
    else:
        lead = _lead_summary(lead_check(v["new"], ctx.ref, v["recent"]))
    early, lag, segs = [], [], []
    q12 = latest_labeled_q(t, 12)
    for q in (q12, q12 - 1):
        c = v["lab12"][v["lab12"]["q"] == q]
        if len(c):
            early.append(_outcome_summary(early_check(c, ctx.emap, ctx.phi12), q))
    q24 = latest_labeled_q(t, HORIZON)
    for q in (q24, q24 - 1):
        c = v["lab24"][v["lab24"]["q"] == q]
        if len(c):
            lag.append(_outcome_summary(lag_check(c, ctx.phi), q))
            segs.append(segment_scan(c, ctx.phi))
    dec = judge(lead if lead.get("schema_ok") else {"schema_ok": False}, early, lag)
    ref = ctx.ref_auc if ref_auc is None else ref_auc
    problem = diagnose(dec["status"], dec["reasons"], lag, ref)
    # 문제 집단: 최신 코호트에서 BH 유의 + 직전 코호트에서도 같은 방향으로 유의하면 '반복 확인'
    seg_now = [s for s in (segs[0] if segs else []) if s["flag"]]
    prev = {(s["group"], s["level"]): s for s in (segs[1] if len(segs) > 1 else []) if s["bh"]}
    for s in seg_now:
        p = prev.get((s["group"], s["level"]))
        s["repeated"] = bool(p and p["rel_to_all"] * s["rel_to_all"] > 0)
    cand = None
    if dec["status"] == "조치 검토":           # 조건부 실행: 문제가 확인된 분기만
        cand = run_candidates(v["lab24"], v["lab12"], v["asof"], ctx.meta, ctx.emap,
                              with_retrain=with_retrain, problem=problem)
    out.update({
        "status": dec["status"], "reasons": dec["reasons"], "problem": problem,
        "problem_label": PROBLEM[problem], "ref_auc": ref,
        "recommendation": recommend(dec["status"], dec["reasons"], cand),
        "evidence": evidence(lead, early, lag, dec["status"], cand) if lag else [],
        "lead": lead, "early": early, "lag": lag,
        "segments": sorted(seg_now, key=lambda s: -abs(s["z"]))[:5],
        "segments_tested": len(segs[0]) if segs else 0,
        "candidates": None if cand is None else {
            "chosen": cand["chosen"], "chosen_label": LABEL.get(cand["chosen"]),
            "residual": cand["residual"], "selection_note": cand["selection_note"],
            "sel_q": cand["sel_q"], "pseudo_asof": cand["pseudo_asof"],
            "windows_pseudo": cand["windows_pseudo"], "windows_deploy": cand["windows_deploy"],
            "table": cand["table"].to_dict("records")},
        "candidate_experiment": "실행" if cand is not None else "생략",
        "approval": "담당자 승인 대기" if cand is not None and cand["chosen"] else None,
    })
    return out, cand


def run_quarter(ctx: Context, t, force=False, with_retrain=True, inject=None) -> dict:
    """inject: 시험용 오류·변화 주입 함수 (view -> view). 실데이터 결과와 구분해 기록"""
    t = pd.Period(t, freq="Q")
    out_dir = RUNS / RULE_VERSION / str(t)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = input_key(ctx, t) + (f"|inject:{inject.__name__}" if inject else "")
    sfx = "" if inject is None else f"_{inject.__name__}"
    dec_path = out_dir / f"decision{sfx}.json"
    if dec_path.exists() and not force:
        old = json.loads(dec_path.read_text(encoding="utf-8"))
        if old.get("input_key") == key and old.get("run_status") == "성공":
            return {**old, "skipped": True}

    t0 = time.time()
    rec = {"quarter": str(t), "input_key": key, "rule_version": RULE_VERSION,
           "model_id": ctx.meta["model_id"], "data_id": ctx.data_id, "code_hash": code_hash(),
           "simulation": inject.__name__ if inject else None,
           "started": pd.Timestamp.now().isoformat(timespec="seconds")}
    try:
        out, cand = decide(ctx, t, with_retrain=with_retrain, inject=inject)
        if cand is not None:
            cand["table"].to_csv(out_dir / f"candidates{sfx}.csv", index=False, encoding="utf-8-sig")
            joblib.dump({"chosen": cand["chosen"], "deploy": cand["deploy"],
                         "deploy_all": cand["deploy_all"], "sel_q": cand["sel_q"]},
                        out_dir / f"candidates{sfx}.joblib")
        rec.update({"run_status": "성공", **out})
    except Exception as e:      # 조용히 잘못된 보고서를 만들지 않음
        rec.update({"run_status": "실패", "status": "판정 유보", "recommendation": RECO["hold"],
                    "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})
    rec["seconds"] = round(time.time() - t0, 1)
    dec_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=_jsonable),
                        encoding="utf-8")
    with open(RUNS / RULE_VERSION / "run_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({k: rec.get(k) for k in
                            ["quarter", "started", "run_status", "status", "recommendation",
                             "candidate_experiment", "simulation", "seconds", "input_key", "error"]},
                           ensure_ascii=False) + "\n")
    return rec
