"""m14. 간단한 결과 화면: app/index.html (+ 저장소 첫 화면 index.html)

분기 선택 → 모형 상태 · 권고 · 이유 3개 · 권고 적용 결과, 예상 부실률 대 실제 부실률 그림.
모든 문장은 runs/<규칙>/<분기>/decision.json 과 사후 평가 결과에서 그대로 만들며, 쉬운 표현만 사용.
상세 분석 화면은 m08_build_app.py 가 app/detail.html 로 만듦.

실행: .venv/Scripts/python modelops/analysis/m14_build_simple.py --rule r4
"""
import argparse
import json
import shutil

import pandas as pd

from config import MO, RESULTS
from m08_build_app import font_faces, kpis, load_decisions, load_json

APP = MO / "app"
STATUS = {"정상": "정상", "주의": "주의", "조치 검토": "조치 필요", "판정 유보": "판단 보류"}
PLAIN = {"maintain": "현행 모형 유지", "recal_intercept": "예측 확률 보정", "recal_platt": "예측 확률 보정",
         "recal_beta": "예측 확률 보정", "recal_isotonic": "예측 확률 보정", "recal_grade": "등급별 예측 확률 보정",
         "recal_early": "예측 확률 보정(1년 실적 반영)", "recal_trend": "예측 확률 보정(추세 반영)",
         "retrain_logit": "최신 자료 재학습(단순 모형)", "retrain_lgbm": "최신 자료 재학습",
         "retrain_lgbm_ext": "최신 자료 재학습(신용정보 변수 추가)",
         "retrain_lgbm_ext_tw": "최신 자료 재학습(최근 대출 가중)"}


def ql(q):
    return f"{q[:4]}년 {q[-1]}분기"


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def gap(oe):
    return f"{abs(oe - 1) * 100:.0f}%"


def reco_text(d):
    c = d.get("candidates") or {}
    if d["status"] == "판정 유보":
        return "자료 확인 전 판단 보류"
    if c.get("chosen"):
        r = PLAIN[c["chosen"]] + " 권고"
        return r + " (적용 후 잔여 오차 존재, 원인 확인 병행)" if c.get("residual") else r
    if d["status"] == "정상":
        return "현행 모형 유지"
    return "모형 변경 보류 및 원인 확인"


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
        hi = "초과" if l0["o_e_ratio"] > 1 else "미달"
        txt = (f"{ql(l0['cohort_q'])} 대출(2년 실적 확정) 실제 부실률 {pct(l0['actual'])}, "
               f"예측 {pct(l0['pred'])}, 예측 대비 {gap(l0['o_e_ratio'])} {hi}")
        e0 = (d.get("early") or [{}])[0]
        if e0:
            txt += f". {ql(e0['cohort_q'])} 대출 1년 실적도 예측 대비 {gap(e0['o_e_ratio'])} {'초과' if e0['o_e_ratio'] > 1 else '미달'}"
        out.append(txt)
    c = d.get("candidates")
    if c is None:
        out.append("오차 기준 미달, 수정안 검증 생략")
    else:
        t = {r["candidate"]: r for r in c["table"]}
        if c["chosen"]:
            k, m = t[c["chosen"]], t["maintain"]
            out.append(f"수정안 {len(t) - 1}종 검증(1년 전 자료로 수립, 이후 실적으로 평가). "
                       f"'{PLAIN[c['chosen']]}' 안 오차 {gap(m['o_e_ratio'])}에서 {gap(k['o_e_ratio'])}로 감소")
        else:
            out.append(f"수정안 {len(t) - 1}종 검증(1년 전 자료로 수립, 이후 실적으로 평가). 기준 충족안 없음")
    return out


def after_text(ph, q):
    rows = [r for r in ph if r["asof_q"] == q and r["part"] == "전체"]
    m = next((r for r in rows if r["candidate"] == "maintain"), None)
    c = next((r for r in rows if r["is_chosen"]), None)
    if not (m and c):
        return None
    return (f"사후 확인. 당시 보유 대출 {c['n']:,}건 적용 시 예측 오차 {gap(m['o_e_ratio'])}에서 "
            f"{gap(c['o_e_ratio'])}로 감소 (실제 부실률 {pct(c['actual'])}, 기존 예측 {pct(m['pred'])}, "
            f"수정 예측 {pct(c['pred'])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="r4")
    ap.add_argument("--default", default="2018Q1")
    a = ap.parse_args()
    decs = load_decisions(a.rule)
    ph = pd.read_csv(RESULTS / f"m06_posthoc_{a.rule}.csv").to_dict("records")
    items = [{"q": d["quarter"], "status": STATUS[d["status"]], "reco": reco_text(d), "why": why_text(d),
              "after": after_text(ph, d["quarter"]),
              "known_until": (d.get("lag") or [{}])[0].get("cohort_q")} for d in decs]
    bt = pd.read_csv(RESULTS / "m03_backtest_quarter.csv")
    bt = bt[(bt.model == "lgbm") & (bt.issue_q >= "2012Q1")]
    cohorts = [{"q": r.issue_q, "actual": r.actual, "pred": r.pred} for r in bt.itertuples()]

    pol = load_json(RESULTS / "m11_policy.json")
    s = {r["policy"][0]: r for r in pol["summary"]}
    d0 = next(r for r in ph if r["asof_q"] == a.default and r["part"] == "전체" and r["candidate"] == "maintain")
    c0 = next(r for r in ph if r["asof_q"] == a.default and r["part"] == "전체" and r["is_chosen"])
    summary = [
        {"label": "이상 감지 시점", "value": f"{int(kpis(decs, ph, a.default)[0]['value'][0]) * 3}개월 단축",
         "note": "대출 1년 실적 병행 확인"},
        {"label": "예측 오차 (권고안 적용 후)", "value": gap(c0["o_e_ratio"]),
         "note": f"적용 전 {gap(d0['o_e_ratio'])}, {ql(a.default)} 보유 대출 {c0['n'] / 1e4:.0f}만 건 기준"},
        {"label": "모형 교체 횟수 (3년)", "value": f"{s['D']['changes']}회",
         "note": f"분기별 재학습 방식 {s['B']['changes']}회 대비, 평균 오차 {s['B']['book_mean_abs_oe_err'] * 100:.1f}% 대비 "
                 f"{s['D']['book_mean_abs_oe_err'] * 100:.1f}%"},
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
