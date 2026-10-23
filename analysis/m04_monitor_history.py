"""m04. 선행 점검·후행 검증 이력 (3단계)

- 선행 점검: 2013Q1~2018Q4 각 분기 신규 실행분 vs 학습 코호트 / 직전 4분기
- 후행 검증: 운영 재현 13분기 각 시점 T 에 확정된 최신 코호트
- 개발 구간(2010~2012) 분기별 과산포 phi 추정 -> 후행 z 보정
- 산출: results/m04_lead.csv, m04_lead_features.csv, m04_lag.csv, m04_lag_groups.csv,
        m04_summary.json, m04_monitor.png

실행: .venv/Scripts/python modelops/analysis/m04_monitor_history.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA, RESULTS, SCORES
from data import load_loans
from labels import LAG, REPORT_LAG
from monitor import EarlyMap, early_check, estimate_phi, lag_check, lead_check


def latest_labeled_q(t: pd.Period, horizon: int) -> pd.Period:
    """판단 분기 t 말에 정답이 분기 전체로 확정된 가장 늦은 실행 분기"""
    m = t.asfreq("M", "end") - (horizon + LAG + REPORT_LAG)
    q = m.asfreq("Q")
    return q if m == q.asfreq("M", "end") else q - 1

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load_scored():
    df = load_loans()
    sc = pd.read_parquet(SCORES, columns=["id", "p_lgbm"])
    df = df.merge(sc, on="id", how="left").rename(columns={"p_lgbm": "p"})
    df["q"] = pd.PeriodIndex(df["issue_d"], freq="Q")
    return df


def main():
    df = load_scored()
    ref = df[(df["issue_d"] >= "2010-01-01") & (df["issue_d"] <= "2012-12-31")]
    phi = estimate_phi(ref)

    lead_rows, feat_rows = [], []
    for q in pd.period_range("2013Q1", "2018Q4", freq="Q"):
        cur = df[df["q"] == q]
        recent = df[(df["q"] >= q - 4) & (df["q"] <= q - 1)]
        r = lead_check(cur, ref, recent)
        f = r["features"].assign(q=str(q))
        feat_rows.append(f)
        top_ref = f.nlargest(3, "psi_ref")
        top_rec = f.nlargest(3, "psi_recent")
        lead_rows.append({
            "q": str(q), "n": r["n"], "volume_vs_recent": r.get("volume_vs_recent"),
            "score_mean": r["score"]["mean_cur"], "score_mean_ref": r["score"]["mean_ref"],
            "score_psi_ref": r["score"]["psi_ref"], "score_psi_recent": r["score"]["psi_recent"],
            "feat_psi_ref_max": f["psi_ref"].max(), "feat_psi_recent_max": f["psi_recent"].max(),
            "n_feat_psi_ref_ge_0.25": int((f["psi_ref"] >= .25).sum()),
            "n_feat_psi_ref_ge_0.10": int((f["psi_ref"] >= .10).sum()),
            "n_feat_psi_recent_ge_0.10": int((f["psi_recent"] >= .10).sum()),
            "top_ref": ", ".join(f"{a}({b:.2f})" for a, b in zip(top_ref.feature, top_ref.psi_ref)),
            "top_recent": ", ".join(f"{a}({b:.2f})" for a, b in zip(top_rec.feature, top_rec.psi_recent)),
            "miss_max_delta_recent_pp": f["miss_delta_recent_pp"].abs().max(),
            "grade_psi_ref": r["mix"]["grade"]["psi_ref"], "grade_psi_recent": r["mix"]["grade"]["psi_recent"],
            "term_psi_recent": r["mix"]["term_m"]["psi_recent"],
            "joint_share": float((cur["application_type"] == "Joint App").mean()),
            "schema_notes": "; ".join(r["schema"]["notes"]),
        })
    lead = pd.DataFrame(lead_rows)
    lead.to_csv(RESULTS / "m04_lead.csv", index=False, encoding="utf-8-sig")
    pd.concat(feat_rows).to_csv(RESULTS / "m04_lead_features.csv", index=False, encoding="utf-8-sig")

    win = json.loads((RESULTS / "m02_summary.json").read_text(encoding="utf-8"))["windows"]
    lag_rows, grp_rows = [], []
    for w in win["replay"]:
        cq = pd.Period(w["lag_check_latest_labeled_issue_q"], freq="Q")
        r = lag_check(df[df["q"] == cq], phi=phi)
        lag_rows.append({"asof_q": w["asof_q"], "cohort_q": str(cq), "n": r["n"],
                         "actual": r["actual"], "pred": r["pred"], "gap_pp": r["gap_pp"],
                         "o_e_ratio": r["o_e_ratio"], "z": r["calib_z"], "z_adj": r["z_adj"],
                         "auc": r["auc"], "auc_lo": r["auc_lo"], "auc_hi": r["auc_hi"],
                         "brier": r["brier"], "calib_a": r["calib_a"], "calib_b": r["calib_b"]})
        for g, t in r["groups"].items():
            grp_rows.append(t.rename(columns={g: "level"}).assign(group=g, asof_q=w["asof_q"]))
    lag = pd.DataFrame(lag_rows)
    lag.to_csv(RESULTS / "m04_lag.csv", index=False, encoding="utf-8-sig")
    grp = pd.concat(grp_rows)
    grp.to_csv(RESULTS / "m04_lag_groups.csv", index=False, encoding="utf-8-sig")
    # ---- 조기 결과 점검 (12개월, 확정 18개월) ----
    emap = EarlyMap().fit(ref["p"], ref["y12"])
    ref = ref.assign(p12=emap(ref["p"]))
    phi12 = estimate_phi(ref, score="p12", y="y12")
    early_rows = []
    for t in pd.period_range("2014Q4", "2018Q4", freq="Q"):
        cq = latest_labeled_q(t, 12)
        r = early_check(df[df["q"] == cq], emap, phi=phi12)
        early_rows.append({"asof_q": str(t), "cohort_q": str(cq), "n": r["n"],
                           "actual12": r["actual"], "pred12": r["pred"], "gap_pp": r["gap_pp"],
                           "o_e_ratio": r["o_e_ratio"], "z_adj": r["z_adj"], "auc": r["auc"]})
    early = pd.DataFrame(early_rows)
    early.to_csv(RESULTS / "m04_early.csv", index=False, encoding="utf-8-sig")

    (RESULTS / "m04_summary.json").write_text(json.dumps(
        {"phi_dev": phi, "phi12_dev": phi12,
         "phi_note": "개발 구간 12분기 mean(z^2). 분기 간 변동이 이항 변동의 phi 배"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    plot(lead, lag, early, RESULTS / "m04_monitor.png")
    print(early.round(4).to_string(index=False))

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 80)
    print("phi", phi)
    print(lead.drop(columns=["top_ref", "top_recent", "schema_notes"]).round(3).to_string(index=False))
    print(lead[["q", "top_ref", "top_recent", "schema_notes"]].to_string(index=False))
    print(lag.round(4).to_string(index=False))
    print(grp[grp.asof_q.isin(["2017Q2", "2018Q4"])].round(4).to_string(index=False))


def plot(lead, lag, early, path):
    qs = [str(q) for q in pd.period_range("2013Q1", "2018Q4", freq="Q")]
    x = {q: i for i, q in enumerate(qs)}
    fig, (a1, a3, a2) = plt.subplots(3, 1, figsize=(10, 8.4), dpi=150, sharex=True)
    xe = [x[q] for q in early.asof_q]
    a3.axhline(0, color="#52514e", lw=.8)
    a3.bar(xe, early["gap_pp"], color=["#e34948" if g > 0 else "#2a78d6" for g in early["gap_pp"]],
           width=.6)
    for xi, g, c in zip(xe, early["gap_pp"], early["cohort_q"]):
        a3.text(xi, g + (.1 if g > 0 else -.1), c, ha="center", fontsize=6.5,
                va="bottom" if g > 0 else "top", color="#52514e")
    a3.set_ylabel("실제 - 기대 (%p)")
    a3.set_title("조기 결과 점검: 12개월 상각, 실행 후 18개월에 확정 (막대 위 = 실행 분기)",
                 fontsize=10, loc="left")
    for a in (a1, a2, a3):
        a.axvspan(x["2015Q4"] - .5, x["2018Q4"] + .5, color="#e8e7e2", zorder=0)
        a.grid(axis="y", color="#e8e7e2", lw=.8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    xl = [x[q] for q in lead.q]
    a1.plot(xl, lead["score_psi_ref"], color="#2a78d6", lw=2, label="점수 분포 PSI (학습 대비)")
    a1.plot(xl, lead["score_psi_recent"], color="#2a78d6", lw=1.4, ls="--", label="점수 분포 PSI (직전 4분기 대비)")
    a1.plot(xl, lead["grade_psi_ref"], color="#eb6834", lw=2, label="등급 구성 PSI (학습 대비)")
    a1.set_ylabel("PSI"); a1.legend(fontsize=8, frameon=False, loc="upper left")
    a1.set_title("선행 점검: 그 분기 신규 실행 대출 (정답 불필요)", fontsize=10, loc="left")
    xg = [x[q] for q in lag.asof_q]
    a2.axhline(0, color="#52514e", lw=.8)
    a2.bar(xg, lag["gap_pp"], color=["#e34948" if g > 0 else "#2a78d6" for g in lag["gap_pp"]],
           width=.6)
    for xi, (g, c) in zip(xg, zip(lag["gap_pp"], lag["cohort_q"])):
        a2.text(xi, g + (.15 if g > 0 else -.15), c, ha="center", fontsize=6.5,
                va="bottom" if g > 0 else "top", color="#52514e")
    a2.set_ylabel("실제 - 예측 (%p)")
    a2.set_title("후행 검증: 판단 시점에 결과가 확정된 최신 코호트 (막대 위 = 실행 분기)", fontsize=10, loc="left")
    a2.set_xticks(range(0, len(qs), 2)); a2.set_xticklabels(qs[::2], fontsize=8)
    a2.set_xlabel("판단 시점 (음영 = 운영 재현 13분기)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
