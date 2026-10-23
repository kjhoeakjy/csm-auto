"""m14. 간단한 결과 화면: app/index.html (+ 저장소 첫 화면 index.html)

분기 선택 → 모형 상태 · 권고 · 이유 3개 · 권고 적용 결과, 예상 상각률 대 실제 상각률 그림.
모든 문장은 runs/<규칙>/<분기>/decision.json 과 사후 평가 결과에서 그대로 만들며, 쉬운 표현만 사용.
상세 분석 화면은 m08_build_app.py 가 app/detail.html 로 만듦.

실행: .venv/Scripts/python modelops/analysis/m14_build_simple.py --rule r4
"""
import argparse
import math
import re
import json
import shutil

import pandas as pd

from config import BACKTEST, MO, RESULTS
from m08_build_app import font_faces, kpis, load_decisions, load_json

APP = MO / "app"
STATUS = {"정상": "정상", "주의": "주의", "조치 검토": "조치 필요", "판정 유보": "판단 보류"}
PLAIN = {"maintain": "현행 모형 유지", "recal_intercept": "예측 상각률 일괄 조정", "recal_platt": "예측 상각률 일괄 조정",
         "recal_beta": "예측 상각률 일괄 조정", "recal_isotonic": "예측 상각률 일괄 조정", "recal_grade": "등급별 예측 상각률 조정",
         "recal_early": "예측 상각률 일괄 조정", "recal_trend": "예측 상각률 일괄 조정",
         "retrain_logit": "최근 대출 자료로 모형 재학습", "retrain_lgbm": "최근 대출 자료로 모형 재학습",
         "retrain_lgbm_ext": "최근 대출 자료로 모형 재학습",
         "retrain_lgbm_ext_tw": "최근 대출 자료로 모형 재학습",
         "recal_early_trend": "1년 중간 실적 추세로 예측 상각률 조정"}


def ql(q):
    return f"{q[:4]}년 {q[-1]}분기"


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def gap(oe):
    return f"{abs(oe - 1) * 100:.0f}%"


def change(a, b):
    """오차 변화 문구. 정수 반올림이 같으면 소수 첫째 자리까지, 증감 방향을 그대로 표기"""
    x, y = abs(a - 1) * 100, abs(b - 1) * 100
    d = 0 if round(x) != round(y) else 1
    if round(x, d) == round(y, d):
        return f"{x:.{d}f}%로 변화 없음"
    return f"{x:.{d}f}%에서 {y:.{d}f}%로 {'감소' if y < x else '증가'}"


def vs_pred(oe):
    """예측 대비 차이 문구 (0%로 반올림되면 일치)"""
    g = abs(oe - 1) * 100
    return "예측과 일치" if round(g) == 0 else f"예측 대비 {g:.0f}% {'초과' if oe > 1 else '미달'}"


def reco_text(d):
    c = d.get("candidates") or {}
    if d["status"] == "판정 유보":
        return "자료 확인 전 판단 보류"
    if c.get("chosen"):
        r = PLAIN[c["chosen"]] + " 권고"
        return r + " (적용 후에도 오차 일부 남음, 원인 확인 병행)" if c.get("residual") else r
    if d["status"] == "정상":
        return "현행 모형 유지"
    return "모형 갱신 보류 및 원인 확인"


def why_text(d):
    out = []
    lead = d.get("lead") or {}
    if lead.get("skipped"):
        out.append("신규 대출 자료 부족, 고객 구성 점검 보류")
    elif lead.get("schema_ok"):
        same = "변화 미미" if lead["score_psi_ref"] < 0.1 else "변화 발생"
        out.append(f"신규 대출 {lead['n']:,}건, 모형 개발 시점 대비 고객 구성 {same}")
    l0 = (d.get("lag") or [{}])[0]
    if l0:
        txt = (f"{ql(l0['cohort_q'])} 대출(2년 결과 확인 완료) 실제 상각률 {pct(l0['actual'])}, "
               f"예측 {pct(l0['pred'])}, {vs_pred(l0['o_e_ratio'])}")
        e0 = (d.get("early") or [{}])[0]
        if e0:
            txt += f". {ql(e0['cohort_q'])} 대출의 1년 시점 상각 결과는 {vs_pred(e0['o_e_ratio'])}"
        out.append(txt)
    c = d.get("candidates")
    if c is None:
        out.append("예측과 실제의 차이가 기준 이내, 수정안 비교 생략")
    else:
        t = {r["candidate"]: r for r in c["table"]}
        if c["chosen"] == "recal_early_trend":
            a = float(re.search(r"logit ([+-][\d.]+)", c["selection_note"]).group(1))
            out.append(f"대출 후 1년 중간 실적 최근 4개 분기의 추세로 현재 보유 대출 수준 추정. 현재 예측 대비 "
                       f"약 {abs(math.exp(a) - 1) * 100:.0f}% {'높음' if a > 0 else '낮음'} (기준 10%), "
                       f"'{PLAIN[c['chosen']]}' 권고")
        elif c["chosen"]:
            k, m = t[c["chosen"]], t["maintain"]
            out.append(f"수정안 {len(t) - 1}종을 1년 전 자료로 만든 뒤 이후 확정된 최근 4개 분기 실적으로 비교 (수정안 선택 단계 검증). "
                       f"'{PLAIN[c['chosen']]}' 안의 오차 {change(m['o_e_ratio'], k['o_e_ratio'])}, 예측 정확도 지표(log loss) {k['rel_ll_gain'] * 100:.1f}% 개선")
        else:
            out.append(f"수정안 {len(t) - 1}종을 1년 전 자료로 만든 뒤 이후 확정된 최근 4개 분기 실적으로 비교 (수정안 선택 단계 검증). 기준 통과안 없음")
    return out


def after_text(ph, q):
    """권고안 사후 평가: 판단 시점 보유 대출에 적용한 예측과 이후 확정된 실제 상각률 (판단에는 미사용)"""
    rows = [r for r in ph if r["asof_q"] == q and r["part"] == "전체"]
    m = next((r for r in rows if r["candidate"] == "maintain"), None)
    c = next((r for r in rows if r["is_chosen"]), None)
    if not (m and c):
        return None
    lo, hi = c["cohorts"].split("~")
    return {"from": lo, "to": hi, "n": int(c["n"]), "actual": c["actual"], "pred_m": m["pred"], "pred_c": c["pred"],
            "text": f"예측 오차(상대) {change(m['o_e_ratio'], c['o_e_ratio'])}, "
                    f"절대 차이 {abs(m['actual'] - m['pred']) * 100:.1f}%p에서 {abs(c['actual'] - c['pred']) * 100:.1f}%p로"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="r5")
    ap.add_argument("--default", default="2018Q1")
    a = ap.parse_args()
    decs = load_decisions(a.rule)
    ph = pd.read_csv(RESULTS / f"m06_posthoc_{a.rule}.csv").to_dict("records")
    items = [{"q": d["quarter"], "status": STATUS[d["status"]], "reco": reco_text(d), "why": why_text(d),
              "after": after_text(ph, d["quarter"]),
              "known_until": (d.get("lag") or [{}])[0].get("cohort_q")} for d in decs]
    bt = pd.read_csv(BACKTEST)
    bt = bt[(bt.model == "lgbm") & (bt.issue_q >= "2012Q1")]
    cohorts = [{"q": r.issue_q, "actual": r.actual, "pred": r.pred} for r in bt.itertuples()]

    pol = load_json(RESULTS / "m11_policy.json")
    s = {r["policy"][0]: r for r in pol["summary"]}
    whole = [r for r in ph if r["part"] == "전체"]
    rec = sorted({r["asof_q"] for r in whole if r["is_chosen"]})           # 수정안 권고 시점
    err = lambda cond: sum(abs(next(r for r in whole if r["asof_q"] == q and cond(r))["o_e_ratio"] - 1)
                           for q in rec) / len(rec) * 100
    e_m, e_c = err(lambda r: r["candidate"] == "maintain"), err(lambda r: r["is_chosen"])
    auc_txt = ("변별력(AUC) 우위" if s["D"]["book_mean_auc"] > s["B"]["book_mean_auc"] + 0.0005
               else "변별력(AUC) 정기 재학습 우위")
    summary = [
        {"label": "상각률 이상 경고 시점", "value": kpis(decs, ph, a.default)[0]["value"],
         "note": "대출 후 1년 중간 실적 활용 (기존 2년 실적), 2016~2017년 단일 사례"},
        {"label": "수정안 적용 후 상각률 예측 오차", "value": f"{e_c:.0f}%",
         "note": f"적용 전 {e_m:.0f}%, 상대 오차 (실제÷예측-1). 권고 {len(rec)}개 시점 평균, 당시 보유 대출 사후 평가"},
        {"label": "3년간 모형 갱신 횟수", "value": f"{s['D']['changes']}회",
         "note": f"예측치 조정 {s['D']['changes'] - s['D']['retrains']}회" + (f", 재학습 {s['D']['retrains']}회" if s["D"]["retrains"] else "") + " (갱신 승인 가정 누적 운영). " +
                 f"정기 재학습 {s['B']['changes']}회 대비 평균 오차 {s['B']['book_mean_abs_oe_err'] * 100:.1f}%에서 "
                 f"{s['D']['book_mean_abs_oe_err'] * 100:.1f}%로. {auc_txt} "
                 f"({s['D']['book_mean_auc']:.3f} 대 {s['B']['book_mean_auc']:.3f})"},
    ]
    data = {"items": items, "cohorts": cohorts, "default_q": a.default, "summary": summary}
    tpl = (APP / "simple.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, default=float)) \
              .replace("/*__FONTS__*/", font_faces())
    (APP / "index.html").write_text(html, encoding="utf-8")
    (MO / "index.html").write_text(html.replace('href="detail.html"', 'href="app/detail.html"'), encoding="utf-8")
    print("index.html", len(html) // 1000, "KB", [x["status"] for x in items])


if __name__ == "__main__":
    main()
