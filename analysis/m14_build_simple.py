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
         "recal_early_trend": "1년 차 실적 추세에 맞춘 예측 상각률 조정"}


def ql(q):
    return f"{q[:4]}년 {q[-1]}분기"


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def gap(oe):
    return f"{abs(oe - 1) * 100:.0f}%"


def change(a, b):
    """오차 변화 문장 (존댓말). 정수 반올림이 같으면 소수 첫째 자리까지"""
    x, y = abs(a - 1) * 100, abs(b - 1) * 100
    d = 0 if round(x) != round(y) else 1
    if round(x, d) == round(y, d):
        return f"{x:.{d}f}%로 변화가 없습니다"
    return f"{x:.{d}f}%에서 {y:.{d}f}%로 {'줄었습니다' if y < x else '늘었습니다'}"


def vs_pred(oe):
    """예측 대비 차이 문장 (존댓말)"""
    g = abs(oe - 1) * 100
    return "예측과 같습니다" if round(g) == 0 else f"예측보다 {g:.0f}% {'높습니다' if oe > 1 else '낮습니다'}"


def eul(w):
    """목적격 조사: 받침 있으면 을, 없으면 를"""
    c = w[-1]
    return w + ("을" if "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 else "를")


def reco_text(d):
    c = d.get("candidates") or {}
    if d["status"] == "판정 유보":
        return "자료를 확인할 때까지 판단을 보류합니다"
    if c.get("chosen"):
        r = f"{eul(PLAIN[c['chosen']])} 권고합니다"
        return r + " (적용 후에도 남는 오차는 원인을 함께 확인합니다)" if c.get("residual") else r
    if d["status"] == "정상":
        return "현행 모형 유지를 권고합니다"
    return "모형 갱신을 보류하고 원인 확인을 권고합니다"


def why_text(d):
    out = []
    lead = d.get("lead") or {}
    if lead.get("skipped"):
        out.append("신규 대출 자료가 부족해 고객 구성 점검을 보류했습니다.")
    elif lead.get("schema_ok"):
        same = "거의 없습니다" if lead["score_psi_ref"] < 0.1 else "있습니다"
        out.append(f"이번 분기 신규 대출은 {lead['n']:,}건입니다. 모형을 만든 시점과 비교해 고객 구성 변화가 {same}.")
    l0 = (d.get("lag") or [{}])[0]
    if l0:
        txt = (f"2년 결과가 확정된 {ql(l0['cohort_q'])} 대출의 실제 상각률은 {pct(l0['actual'])}, "
               f"예측은 {pct(l0['pred'])}로 {vs_pred(l0['o_e_ratio'])}.")
        e0 = (d.get("early") or [{}])[0]
        if e0:
            txt += f" {ql(e0['cohort_q'])} 대출의 1년 차 상각률은 {vs_pred(e0['o_e_ratio'])}."
        out.append(txt)
    c = d.get("candidates")
    if c is None:
        out.append("예측과 실제의 차이가 기준 이내여서 수정안 비교를 생략했습니다.")
    else:
        t = {r["candidate"]: r for r in c["table"]}
        if c["chosen"] == "recal_early_trend":
            a = float(re.search(r"logit ([+-][\d.]+)", c["selection_note"]).group(1))
            out.append(f"최근 4개 분기 대출의 1년 차 상각률 추세로 보면 결과가 아직 나오지 않은 대출의 상각률이 "
                       f"현재 예측보다 약 {abs(math.exp(a) - 1) * 100:.0f}% {'높을' if a > 0 else '낮을'} 것으로 추정됩니다. "
                       f"기준(10%)을 넘어 예측 상각률 조정을 권고합니다.")
        elif c["chosen"]:
            k, m = t[c["chosen"]], t["maintain"]
            out.append(f"수정안 {len(t) - 1}종을 1년 전 자료로 만들어 이후 실제 결과와 비교했습니다. "
                       f"'{PLAIN[c['chosen']]}' 안의 오차가 {change(m['o_e_ratio'], k['o_e_ratio'])}.")
        else:
            out.append(f"수정안 {len(t) - 1}종을 1년 전 자료로 만들어 이후 실제 결과와 비교했지만 기준을 통과한 안이 없습니다.")
    return out


def after_text(ph, q):
    """권고안을 당시 보유 대출에 적용한 예측과 이후 확정된 실제 상각률 비교 (판단에는 미사용)"""
    rows = [r for r in ph if r["asof_q"] == q and r["part"] == "전체"]
    m = next((r for r in rows if r["candidate"] == "maintain"), None)
    c = next((r for r in rows if r["is_chosen"]), None)
    if not (m and c):
        return None
    lo, hi = c["cohorts"].split("~")
    return {"from": lo, "to": hi, "n": int(c["n"]), "actual": c["actual"], "pred_m": m["pred"], "pred_c": c["pred"],
            "text": f"예측 오차가 {change(m['o_e_ratio'], c['o_e_ratio'])}."}


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
    pq = pd.read_csv(RESULTS / "m11_policy_quarter.csv")
    last_q = pq.quarter.iloc[-1]
    oe_last = {r["policy"][0]: r["book_oe"] for r in pq[pq.quarter == last_q].to_dict("records")}
    auc_txt = ("변별력(AUC) 우위" if s["D"]["book_mean_auc"] > s["B"]["book_mean_auc"] + 0.0005
               else "변별력(AUC) 정기 재학습 우위")
    summary = [
        {"label": "상각률 이상 경고 시점", "value": kpis(decs, ph, a.default)[0]["value"],
         "note": "대출 1년 차 상각률로 경고합니다. 기존 방식은 2년 차 결과를 확인한 뒤 경고합니다."},
        {"label": "모형 갱신 후 상각률 예측 오차", "value": f"{abs(oe_last['D'] - 1) * 100:.0f}%",
         "note": f"모형을 갱신하지 않으면 예측 오차가 {abs(oe_last['A'] - 1) * 100:.0f}%입니다 ({ql(last_q)} 기준)."},
        {"label": "3년간 모형 갱신 횟수", "value": f"{s['D']['changes']}회",
         "note": f"3년 평균 예측 오차는 {s['D']['book_mean_abs_oe_err'] * 100:.1f}%입니다. 3개월마다 재학습하면 "
                 f"{s['B']['changes']}회 갱신하고도 예측 오차가 {s['B']['book_mean_abs_oe_err'] * 100:.1f}%입니다."},
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
