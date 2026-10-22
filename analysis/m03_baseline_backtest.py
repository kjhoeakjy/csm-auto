"""m03. 기준모형 학습(2010~2012) + 시간외 검증 + 분기별 백테스트 (2단계)

- 로지스틱(기준선)과 LightGBM 을 같은 입력(CORE)으로 학습
  적합 2010Q1~2012Q2 / 선택(튜닝) 2012Q3~Q4 / 최종 = 둘을 합쳐 재적합
- 2013Q1~2016Q3 실행 코호트(정답 확정)에서 분기별 예측 대비 실제, 판별력, 보정 지표
- 운영 재현 13분기 각 시점에서 '그때 볼 수 있었던' 후행 검증 코호트의 결과 정리
- 산출: results/m03_*.csv|json|png, data/scores_v1.parquet, results/models/champion_v1.joblib

실행: .venv/Scripts/python modelops/analysis/m03_baseline_backtest.py
"""
import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import features as F
from config import DATA, RESULTS, SEED
from data import load_loans
from metrics import by_group, evaluate
from models import fit_lgbm, fit_logit

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
MODELS = RESULTS / "models"
MODELS.mkdir(exist_ok=True)
FIT = ("2010-01-01", "2012-06-30")
SEL = ("2012-07-01", "2012-12-31")
OOT = ("2013-01-01", "2016-09-30")      # 정답 완전한 마지막 실행 분기 2016Q3


def code_hash():
    h = hashlib.sha1()
    for f in ["features.py", "labels.py", "models.py", "metrics.py", "m03_baseline_backtest.py"]:
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:12]


def between(df, a, b):
    return (df["issue_d"] >= a) & (df["issue_d"] <= b)


def main():
    t0 = time.time()
    df = load_loans()
    X = F.build(df)
    num, cat = F.num_cols(), F.CAT_COLS
    fit, sel = between(df, *FIT), between(df, *SEL)

    logit, logit_info = fit_logit(X[fit], df.loc[fit, "y"], X[sel], df.loc[sel, "y"], num, cat)
    lgbm, lgbm_info = fit_lgbm(X.loc[fit, num + cat], df.loc[fit, "y"],
                               X.loc[sel, num + cat], df.loc[sel, "y"])
    df["p_logit"] = logit.predict_proba(X[num + cat])[:, 1]
    df["p_lgbm"] = lgbm.predict_proba(X[num + cat])[:, 1]

    meta = {
        "model_id": "champion_v1", "trained_on": "2010Q1~2012Q4 실행 (적합 ~2012Q2, 선택 2012Q3~Q4)",
        "train_n": int((fit | sel).sum()), "train_rate": float(df.loc[fit | sel, "y"].mean()),
        "label": "STOP24", "features": num + cat, "seed": SEED, "code_hash": code_hash(),
        "logit": {k: (v if k != "sel_logloss" else {str(a): b for a, b in v.items()})
                  for k, v in logit_info.items()},
        "lgbm": {"params": lgbm_info["params"], "best_iter": lgbm_info["best_iter"],
                 "n_estimators": lgbm_info["n_estimators"],
                 "sel_logloss": [[r[0], r[1], str(r[2])] for r in lgbm_info["sel_logloss"]]},
    }
    joblib.dump({"logit": logit, "lgbm": lgbm, "meta": meta}, MODELS / "champion_v1.joblib")
    df[["id", "issue_d", "issue_q", "y", "label_known_d", "grade", "term_m", "purpose",
        "p_logit", "p_lgbm"]].to_parquet(DATA / "scores_v1.parquet", index=False)

    # ---- 시간외 검증 ----
    oot = df[between(df, *OOT)]
    overall = {m: evaluate(oot["y"], oot[f"p_{m}"]) for m in ("logit", "lgbm")}
    insample = {m: evaluate(df.loc[fit | sel, "y"], df.loc[fit | sel, f"p_{m}"])
                for m in ("logit", "lgbm")}
    rows = []
    for m in ("logit", "lgbm"):
        t = by_group(df[between(df, "2010-01-01", OOT[1])], "issue_q", p=f"p_{m}")
        t["model"] = m
        rows.append(t)
    bt = pd.concat(rows)
    bt.to_csv(RESULTS / "m03_backtest_quarter.csv", index=False, encoding="utf-8-sig")

    oot = oot.assign(year=oot["issue_d"].dt.year)
    gr = by_group(oot, ["year", "grade"], p="p_lgbm")
    gr[["year", "grade"]] = pd.DataFrame(gr["group"].tolist(), index=gr.index)
    gr.drop(columns="group").to_csv(RESULTS / "m03_grade_year.csv", index=False,
                                    encoding="utf-8-sig")
    seg = pd.concat([by_group(oot, "term_m", p="p_lgbm").assign(seg="term"),
                     by_group(oot, "purpose", p="p_lgbm").assign(seg="purpose")])
    seg.to_csv(RESULTS / "m03_segment.csv", index=False, encoding="utf-8-sig")

    # ---- 운영 재현 시점별 후행 검증 (그 시점에 확정된 최신 실행 분기) ----
    win = json.loads((RESULTS / "m02_summary.json").read_text(encoding="utf-8"))["windows"]
    q_l = bt[bt["model"] == "lgbm"].set_index("issue_q")
    q_g = bt[bt["model"] == "logit"].set_index("issue_q")
    replay = []
    for r in win["replay"]:
        q = r["lag_check_latest_labeled_issue_q"]
        a, b = q_l.loc[q], q_g.loc[q]
        replay.append({"asof_q": r["asof_q"], "cohort_q": q, "n": int(a["n"]),
                       "actual": a["actual"], "pred_lgbm": a["pred"], "gap_pp_lgbm": a["gap_pp"],
                       "z_lgbm": a["calib_z"], "auc_lgbm": a["auc"], "pred_logit": b["pred"],
                       "gap_pp_logit": b["gap_pp"], "auc_logit": b["auc"]})
    rp = pd.DataFrame(replay)
    rp.to_csv(RESULTS / "m03_replay_lagcheck.csv", index=False, encoding="utf-8-sig")

    summary = {"meta": meta, "insample": insample, "oot_overall": overall,
               "seconds": round(time.time() - t0, 1)}
    (RESULTS / "m03_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    plot(bt, rp, RESULTS / "m03_backtest.png")

    pd.set_option("display.width", 220, "display.max_columns", 30)
    cols = ["issue_q", "n", "actual", "pred", "gap_pp", "calib_z", "auc", "calib_a", "calib_b"]
    print("[LGBM]\n", bt[bt.model == "lgbm"][cols].round(4).to_string(index=False))
    print("[LOGIT]\n", bt[bt.model == "logit"][cols].round(4).to_string(index=False))
    print(rp.round(4).to_string(index=False))
    print(json.dumps({"insample": insample, "oot": overall}, indent=1, default=float))
    print("logit", logit_info["C"], "lgbm", lgbm_info["params"], lgbm_info["best_iter"])


def plot(bt, rp, path):
    l = bt[bt.model == "lgbm"].reset_index(drop=True)
    g = bt[bt.model == "logit"].reset_index(drop=True)
    x = np.arange(len(l))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), dpi=150, sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    tr = l.index[l.issue_q <= "2012Q4"]
    for a in (ax, ax2):
        a.axvspan(tr.min() - .5, tr.max() + .5, color="#e8e7e2", zorder=0)
        a.grid(axis="y", color="#e8e7e2", lw=.8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax.fill_between(x, l["actual_lo"] * 100, l["actual_hi"] * 100, color="#0b0b0b", alpha=.10, lw=0)
    ax.plot(x, l["actual"] * 100, color="#0b0b0b", lw=2, label="실제 상각률 (95% 구간)")
    ax.plot(x, l["pred"] * 100, color="#2a78d6", lw=2, label="예측 평균: LightGBM")
    ax.plot(x, g["pred"] * 100, color="#eb6834", lw=1.6, ls="--", label="예측 평균: 로지스틱")
    ax.text(np.mean(tr), ax.get_ylim()[1] * .97, "학습 코호트", ha="center", fontsize=8,
            color="#52514e", va="top")
    ax.set_ylabel("24개월 상각률 (%)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("실행 분기별 예측 대비 실제 (학습 2010~2012, 이후는 시간외)", fontsize=10, loc="left")
    ax2.plot(x, l["auc"], color="#2a78d6", lw=2, label="LightGBM")
    ax2.plot(x, g["auc"], color="#eb6834", lw=1.6, ls="--", label="로지스틱")
    ax2.set_ylabel("AUC"); ax2.legend(fontsize=8, frameon=False, loc="lower right")
    ax2.set_xticks(x[::4]); ax2.set_xticklabels(l["issue_q"].values[::4], fontsize=8)
    ax2.set_xlabel("대출 실행 분기 (결과 확정 = 실행 후 30개월)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
