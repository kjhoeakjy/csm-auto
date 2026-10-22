"""m08. 결과 화면 생성: modelops/app/index.html (단일 파일, 데이터·글꼴 내장, 인터넷 불필요)

입력: runs/<규칙>/<분기>/decision.json, results/m03_backtest_quarter.csv, results/m06_posthoc_<규칙>.csv,
      runs/<규칙>/run_log.jsonl, results/m07_robustness.csv, runs/r1/*/decision.json
보고서는 report.build 로 만들고 report.verify 로 수치 대조. 대조 실패 시 화면에 '발행 보류' 표시

실행: .venv/Scripts/python modelops/analysis/m08_build_app.py --rule r2.1 --default 2018Q1
"""
import argparse
import base64
import json

import numpy as np
import pandas as pd

import report
from config import MO, RESULTS, ROOT
from pipeline import RUNS

APP = MO / "app"
FONTS = ROOT / "app" / "fonts"          # 이전 프로젝트 글꼴(OFL) 재사용, 읽기만 함


def font_faces():
    css = []
    for name, w in [("Regular", 400), ("Medium", 500), ("Bold", 700)]:
        b = base64.b64encode((FONTS / f"Pretendard-{name}.subset.woff2").read_bytes()).decode()
        css.append(f'@font-face{{font-family:"Pretendard";font-weight:{w};font-display:swap;'
                   f'src:url(data:font/woff2;base64,{b}) format("woff2")}}')
    return "\n".join(css)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k not in ("trace",)}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


def load_decisions(rule):
    out = []
    for d in sorted((RUNS / rule).iterdir()):
        f = d / "decision.json"
        if d.is_dir() and f.exists():
            out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


def load_llm(model="claude-opus-5-5"):
    p = RESULTS / f"m09_llm_report_{model}.jsonl"
    if not p.exists():
        return None
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["quarter"]] = {"published": r["published"], "problems": r["problems"], "model": r["model"],
                             "seconds": r["seconds"], "output": r["output"] if r["published"] else None}
    return out


def load_json(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def timing_summary():
    p = RESULTS / "m12_label_timing.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    w = d.pivot(index="quarter", columns="lag", values="status")
    r = d.pivot(index="quarter", columns="lag", values="recommendation")
    same_s = int(((w[4] == w[5]) & (w[6] == w[5])).sum())
    same_r = int(((r[4] == r[5]) & (r[6] == r[5])).sum())
    diff = [q for q in w.index if not (w.loc[q, 4] == w.loc[q, 5] == w.loc[q, 6])]
    return (f"13개 분기 중 상태 일치 {same_s}개, 권고 일치 {same_r}개 (재학습 제외, 재보정 후보만 비교)."
            + (f" 상태가 달라진 분기: {', '.join(diff)}" if diff else ""))


def kpis(decs, posthoc, default_q):
    out = []
    first = lambda pred: next((d["quarter"] for d in decs if pred(d)), None)
    e = first(lambda d: any(r.startswith("early12") for r in d.get("reasons", []))
              and d.get("early") and d["early"][0]["o_e_ratio"] > 1)
    l = first(lambda d: "lag24_material_persistent" in d.get("reasons", [])
              and d.get("lag") and d["lag"][0]["o_e_ratio"] > 1)
    if e and l:
        gap = pd.Period(l, freq="Q").ordinal - pd.Period(e, freq="Q").ordinal
        out.append({"label": "조기 경고", "value": f"{gap}분기 먼저",
                    "note": f"과소 예측을 12개월 중간 신호로 {e}에 포착. 24개월 확정 결과로는 {l}에 확인"})
    ph = [r for r in posthoc if r["asof_q"] == default_q and r["part"] == "전체"]
    m = next((r for r in ph if r["candidate"] == "maintain"), None)
    c = next((r for r in ph if r["is_chosen"]), None)
    if m and c:
        out.append({"label": "예측 오차 (사후 확인)",
                    "value": f"{abs(m['o_e_ratio'] - 1) * 100:.0f}% → {abs(c['o_e_ratio'] - 1) * 100:.0f}%",
                    "note": f"{default_q} 권고({c['label']})를 판단 당시 보유 대출 {c['n']:,}건에 적용. "
                            f"AUC {m['auc']:.3f} → {c['auc']:.3f}"})
    pol = load_json(RESULTS / "m11_policy.json")
    if pol:
        s = {r["policy"][0]: r for r in pol["summary"]}
        if all(k in s for k in "ABD"):
            out.append({"label": "모형 변경 (13개 분기)",
                        "value": f"{s['B']['changes']}회 → {s['D']['changes']}회",
                        "note": f"매 분기 재학습 대비. 예측 오차는 비슷 ({s['B']['book_mean_abs_oe_err'] * 100:.1f}% vs "
                                f"{s['D']['book_mean_abs_oe_err'] * 100:.1f}%, 고정 모형 {s['A']['book_mean_abs_oe_err'] * 100:.1f}%)"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="r3")
    ap.add_argument("--default", default="2018Q1")
    a = ap.parse_args()
    decs = load_decisions(a.rule)
    quarters, asof = [], []
    for d in decs:
        txt = report.build(d)
        chk = report.verify(txt, d) if txt else {"verified": False, "n_numbers": 0, "unmatched": []}
        card = report.card_evidence(d)
        card_chk = report.verify(" ".join(e["text"] + " " + e.get("short", "") for e in card), d)
        if not card_chk["verified"]:           # 대조 실패 시 정량 근거 원문 사용
            card = d.get("evidence", [])
        lag0 = (d.get("lag") or [{}])[0]
        e0 = (d.get("early") or [{}])[0]
        quarters.append({**{k: d.get(k) for k in ["quarter", "asof", "status", "recommendation", "reasons",
                                                   "evidence", "lead", "early", "lag", "candidates",
                                                   "run_status", "candidate_experiment", "problem",
                                                   "problem_label", "ref_auc", "segments",
                                                   "segments_tested"]},
                         "q": d["quarter"], "lag_cohort": lag0.get("cohort_q"), "card": card,
                         "card_check": card_chk,
                         "report": txt if chk["verified"] else "", "report_check": chk})
        asof.append({"q": d["quarter"], "status": d["status"], "lag_q": lag0.get("cohort_q"),
                     "lag_oe": lag0.get("o_e_ratio"), "early_q": e0.get("cohort_q"),
                     "early_oe": e0.get("o_e_ratio")})

    bt = pd.read_csv(RESULTS / "m03_backtest_quarter.csv")
    l = bt[bt.model == "lgbm"].set_index("issue_q")
    g = bt[bt.model == "logit"].set_index("issue_q")
    cohort = [{"q": q, "n": int(r["n"]), "actual": r["actual"], "pred": r["pred"],
               "pred_logit": g.loc[q, "pred"], "lo": r["actual_lo"], "hi": r["actual_hi"]}
              for q, r in l.iterrows()]

    ph_path = RESULTS / f"m06_posthoc_{a.rule}.csv"
    posthoc = pd.read_csv(ph_path).to_dict("records") if ph_path.exists() else []

    log = [json.loads(x) for x in (RUNS / a.rule / "run_log.jsonl").read_text(encoding="utf-8").splitlines()]
    last = {}
    for r in log:
        if r.get("simulation") is None:
            last[r["quarter"]] = r
    runlog = [last[k] for k in sorted(last)]

    rob = pd.read_csv(RESULTS / "m07_robustness.csv").to_dict("records")
    r1 = {d["quarter"]: d for d in load_decisions("r1")}
    comp = [{"q": d["quarter"], "r1_status": r1.get(d["quarter"], {}).get("status"),
             "r1_reco": r1.get(d["quarter"], {}).get("recommendation"),
             "reco": d["recommendation"]} for d in decs]

    data = clean({
        "lede": "신규 대출의 24개월 내 상각 위험을 예측하는 모형 1개를 매 분기 자동 점검하고, 문제 발생 시 "
                "유지·재보정·재학습 중 당시 데이터로 검증된 조치를 담당자에게 권고하는 시스템",
        "feat_ko": report.FEAT_KO, "kpis": kpis(decs, posthoc, a.default),
        "policy": load_json(RESULTS / "m11_policy.json"),
        "injection": pd.read_csv(RESULTS / "m10_injection.csv").to_dict("records")
        if (RESULTS / "m10_injection.csv").exists() else None,
        "timing": timing_summary(), "llm": load_llm(),
        "meta": {"rule_version": a.rule, "model_id": decs[0]["model_id"], "data_id": decs[0]["data_id"],
                 "code_hash": decs[0]["code_hash"]},
        "default_q": a.default, "quarters": quarters,
        "trend": {"cohort": cohort, "asof": asof}, "posthoc": posthoc, "runlog": runlog,
        "robustness": rob, "rules_compare": comp,
    })
    tpl = (APP / "template.html").read_text(encoding="utf-8")
    js = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    html = tpl.replace("/*__DATA__*/null", js).replace("/*__FONTS__*/", font_faces())
    (APP / "index.html").write_text(html, encoding="utf-8")
    bad = [q["q"] for q in quarters if q["report"] and not q["report_check"]["verified"]]
    print(f"index.html {len(html) / 1e6:.1f}MB, quarters {len(quarters)}, 보고서 대조 실패 {bad}")


if __name__ == "__main__":
    main()
