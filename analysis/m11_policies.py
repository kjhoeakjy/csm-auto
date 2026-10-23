"""m11. 운영 정책 4종 비교 (과거 분기 재현, 결정은 as-of 데이터만, 평가는 이후 확정 결과)

  A 고정 모형       : 2010~2012 학습 모형을 끝까지 사용
  B 정기 재학습     : 매 분기 최신 확정 8분기로 LightGBM 재학습
  C 단순 임계값     : 운영 중 모형의 최신 확정 코호트 실제/예측이 10% 이상 벗어나면 재학습 (B와 같은 방식)
  D 제안 시스템     : pipeline.decide (판정 → 문제 유형 진단 → 후보 모의 배포 검증 → 선택). 선택 시 승인 가정, 결정 직후 적용 (B·C 도 같은 시점)

평가 (각 판단 분기 T 의 결정 직후 모형으로):
  - 보유 대출: T 시점에 24개월 결과가 미확정이던 코호트(마지막 확정 다음 ~ min(T, 2016Q3))의 예상 상각률
  - 신규 대출: T 에 실행된 대출 (정답이 있는 2016Q3 까지)
지표: 실제/예측 오차 |O/E-1|, log loss, AUC, 모형 변경 횟수, 재학습 횟수
산출: results/m11_policy_quarter.csv, m11_policy_summary.csv, m11_policy.json

실행: .venv/Scripts/python modelops/analysis/m11_policies.py
"""
import json
import time

import numpy as np
import pandas as pd

import features as F
from candidates import KIND, LABEL, Retrain
from config import RESULTS
from labels import HORIZON
from metrics import evaluate
from monitor import EarlyMap
from pipeline import Context, asof_month, decide, latest_labeled_q
from rules import R

QUARTERS = list(pd.period_range("2015Q4", "2018Q4", freq="Q"))
LAST_LABELED = pd.Period("2016Q3", freq="Q")


class Policy:
    def __init__(self, name, p0):
        self.name, self.p, self.changes, self.retrains, self.log = name, p0.copy(), 0, 0, []
        self.model_label = "최초 모형"
        self.ref_auc = None


def retrain_lgbm(ctx, X, asof):
    """as-of 확정 24개월 정답의 최근 8분기로 LightGBM 재학습 → 전체 대출 점수"""
    lab = ctx.df[ctx.df["label_known_d"] <= asof]
    qs = sorted(lab["q"].unique())[-R["retrain_window_quarters"]:]
    d = lab[lab["q"].isin(qs)]
    m = Retrain("retrain_lgbm", ctx.meta).fit(d)
    return m.m_.predict_proba(X[m.cols_])[:, 1]


def apply_chosen(name, model, ctx, X, cur_p):
    if KIND[name] == "recal":
        return model.predict(cur_p, ctx.df["grade"].values)
    return model.m_.predict_proba(X[model.cols_])[:, 1]


def evaluate_policy(ctx, p, t):
    q24 = latest_labeled_q(t, HORIZON)
    end = min(t, LAST_LABELED)
    out = {}
    m = (ctx.df["q"] > q24) & (ctx.df["q"] <= end)
    if m.sum() > 5000:
        e = evaluate(ctx.df.loc[m, "y"], p[m.values])
        out.update({"book_n": e["n"], "book_oe": e["o_e_ratio"], "book_ll": e["logloss"],
                    "book_auc": e["auc"], "book_brier": e["brier"]})
    if t <= LAST_LABELED:
        m = ctx.df["q"] == t
        e = evaluate(ctx.df.loc[m, "y"], p[m.values])
        out.update({"new_n": e["n"], "new_oe": e["o_e_ratio"], "new_ll": e["logloss"], "new_auc": e["auc"]})
    return out


def set_active(ctx, p, refit_emap=False):
    """D 정책의 운영 모형 교체: 점검·후보 엔진이 보는 점수를 교체.
    조기 기대 함수(24개월 확률 -> 12개월 확률)는 재학습 때만 개발 구간에서 재적합. 수준 재보정 뒤 재적합하면
    개발 구간 수준으로 다시 맞춰져 보정 효과가 사라지고 같은 이동이 반복 적용되므로 유지"""
    ctx.df["p"] = p
    rw = ctx.meta.get("ref_window", ["2010-01-01", "2012-12-31"])
    ctx.ref = ctx.df[(ctx.df["issue_d"] >= rw[0]) & (ctx.df["issue_d"] <= rw[1])]
    if refit_emap:
        ctx.emap = EarlyMap().fit(ctx.ref["p"], ctx.ref["y12"])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="예: D (해당 정책만 다시 계산, 나머지는 기존 결과 사용)")
    args = ap.parse_args()
    t_start = time.time()
    ctx = Context.load()
    emap0 = ctx.emap
    X = F.build(ctx.df, extended=True)
    p0 = ctx.df["p"].values.copy()
    names = ["A 고정 모형", "B 정기 재학습", "C 단순 임계값", "D 제안 시스템"]
    if args.only:
        names = [n for n in names if n[0] in args.only]
    pols = {k: Policy(k, p0) for k in names}
    pols_emap = {k: emap0 for k in names}
    rows = []
    for t in QUARTERS:
        asof = asof_month(t)
        q24 = latest_labeled_q(t, HORIZON)
        lab_last = ctx.df["q"] == q24
        for name, pol in pols.items():
            action = "유지"
            if name.startswith("B"):
                pol.p = retrain_lgbm(ctx, X, asof)
                pol.changes += 1; pol.retrains += 1; action = "재학습"
            elif name.startswith("C"):
                oe = ctx.df.loc[lab_last, "y"].sum() / pol.p[lab_last.values].sum()
                if abs(oe - 1) >= R["lag_rel_act"]:
                    pol.p = retrain_lgbm(ctx, X, asof)
                    pol.changes += 1; pol.retrains += 1; action = f"재학습 (실제/예측 {oe:.2f})"
                else:
                    action = f"유지 (실제/예측 {oe:.2f})"
            elif name.startswith("D"):
                set_active(ctx, pol.p)
                ctx.emap = pols_emap[name]
                out, cand = decide(ctx, t, with_retrain=True, ref_auc=pol.ref_auc)
                action = f"{out['status']} / {out['recommendation']}"
                if cand is not None and cand["chosen"]:
                    ch = cand["chosen"]
                    pol.p = apply_chosen(ch, cand["deploy"], ctx, X, pol.p)
                    pol.changes += 1
                    pol.retrains += KIND[ch] == "retrain"
                    if KIND[ch] == "retrain":           # 재학습이면 조기 기대 함수도 새 모형 기준으로 재적합
                        set_active(ctx, pol.p, refit_emap=True)
                        pols_emap[name] = ctx.emap
                    pol.model_label = LABEL[ch]
                    pol.ref_auc = float(cand["table"].set_index("candidate").loc[ch, "auc"])
                    action += f" → 승인 가정, {LABEL[ch]} 적용"
                pol.log.append({"quarter": str(t), "status": out["status"], "problem": out["problem_label"],
                                "recommendation": out["recommendation"],
                                "chosen": LABEL.get(cand["chosen"]) if cand is not None and cand["chosen"] else None,
                                "note": cand["selection_note"] if cand is not None else None})
            ev = evaluate_policy(ctx, pol.p, t)
            rows.append({"quarter": str(t), "policy": name, "action": action, "changes_cum": pol.changes,
                         "retrains_cum": pol.retrains, **ev})
            print(t, name, action, {k: round(v, 4) for k, v in ev.items() if isinstance(v, float)}, flush=True)
        ctx.df["p"] = p0          # 다른 정책에 영향 없도록 원래 점수로 복구
        ctx.emap = emap0

    q = pd.DataFrame(rows)
    if args.only:                  # 다시 계산하지 않은 정책은 기존 결과 유지
        old = pd.read_csv(RESULTS / "m11_policy_quarter.csv")
        q = pd.concat([old[~old["policy"].isin(names)], q]).sort_values(["quarter", "policy"]).reset_index(drop=True)
    q.to_csv(RESULTS / "m11_policy_quarter.csv", index=False, encoding="utf-8-sig")
    summ = []
    for name, g in q.groupby("policy"):
        b = g.dropna(subset=["book_oe"])
        n = g.dropna(subset=["new_oe"])
        summ.append({"policy": name, "changes": int(g["changes_cum"].max()), "retrains": int(g["retrains_cum"].max()),
                     "book_mean_abs_oe_err": float((b["book_oe"] - 1).abs().mean()),
                     "book_mean_ll": float(b["book_ll"].mean()), "book_mean_auc": float(b["book_auc"].mean()),
                     "new_mean_abs_oe_err": float((n["new_oe"] - 1).abs().mean()) if len(n) else None,
                     "new_mean_ll": float(n["new_ll"].mean()) if len(n) else None,
                     "quarters_evaluated": int(len(b))})
    s = pd.DataFrame(summ)
    s.to_csv(RESULTS / "m11_policy_summary.csv", index=False, encoding="utf-8-sig")
    (RESULTS / "m11_policy.json").write_text(json.dumps(
        {"summary": summ, "d_log": pols["D 제안 시스템"].log, "seconds": round(time.time() - t_start)},
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
