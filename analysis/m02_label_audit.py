"""m02. 정답 정의·관측 시점·변수 가용성·누수 검증 (1단계)

만드는 것 (modelops/results/m02_*):
  m02_cohort_labels.csv     실행 분기별 표본 수와 정답 정의별 상각률
  m02_sensitivity.csv       기간별(학습/평가) 정답 정의 민감도 + 대출 단위 불일치율 + 등급 AUC
  m02_final36.csv           36개월 만기 최종 상각 대비 24개월 정답의 포착률
  m02_feature_avail.csv     변수별·실행 연도별 결측률, 제공 개시 연도
  m02_policy_drift.csv      연도별 심사 기준·상품 구성 변화
  m02_lag_evidence.json     상환 중단 -> 상각 지연 근거
  m02_windows.json          조정된 학습·운영 재현 구간
  m02_label_rates.png       분기별 상각률(정답 정의별)

실행: .venv/Scripts/python modelops/analysis/m02_label_audit.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import BASE_PARQUET, ORIG_NUMERIC, ORIG_CATEGORICAL, RESULTS, SNAPSHOT
from labels import HORIZON, LAG, REPORT_LAG, add_stop, make_label, month_diff

SNAP = pd.Timestamp(SNAPSHOT)
TRAIN = ("2010-01-01", "2012-12-31")
EVAL = ("2013-01-01", "2016-06-30")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load():
    cols = (["id", "issue_d", "issue_q", "loan_status", "last_pymnt_d", "policy_flag",
             "debt_settlement_flag", "debt_settlement_flag_date", "settlement_date",
             "hardship_start_date", "term_m", "grade_num", "sub_grade", "grade",
             "application_type", "verification_status", "fico", "loan_amnt", "dti",
             "int_rate", "annual_inc"]
            + [c for c in ORIG_NUMERIC if c not in ("term_m", "grade_num", "fico",
                                                    "loan_amnt", "dti", "int_rate",
                                                    "annual_inc", "emp_years",
                                                    "credit_hist_m")]
            + ["emp_years", "credit_hist_m", "home_ownership", "purpose",
               "initial_list_status", "addr_state"])
    cols = list(dict.fromkeys(cols))
    df = pd.read_parquet(BASE_PARQUET, columns=cols)
    n_all = len(df)
    df = df[df["policy_flag"] == 1].copy()
    return df, n_all


def add_labels(df):
    lab = {}
    lab["stop24"] = make_label(df, method="stop")                       # 기본
    for L in (4, 5, 6):
        lab[f"co24_L{L}"] = make_label(df, method="co", lag=L)
    lab["stop24_settle_adj"] = make_label(df, method="stop", settlement="adjust")
    for k, v in lab.items():
        df[k] = v
    df["settled"] = (df["debt_settlement_flag"] == "Y").astype(int)
    d = add_stop(df[["issue_d", "last_pymnt_d", "debt_settlement_flag",
                     "debt_settlement_flag_date"]].copy())
    df["stop_m"] = d["stop_m"]
    # 스냅샷 시점에 아직 상각되지 않았지만 24개월 안에 상환이 끊긴 대출 = 정답 미확정
    open_status = ~df["loan_status"].isin(["Charged Off", "Fully Paid"])
    df["pending24"] = (open_status & (df["stop_m"] <= HORIZON)
                       & (month_diff(pd.Series(SNAP, index=df.index), d["stop_d"]) >= 1)).astype(int)
    # 36개월 만기 대출의 최종 상각 여부
    df["final_co"] = df["loan_status"].eq("Charged Off").astype(int)
    return df, list(lab)


def cohort_table(df, labs):
    g = df.groupby("issue_q")
    t = pd.DataFrame({
        "n": g.size(), "share_60m": g["term_m"].apply(lambda s: (s == 60).mean()),
        "share_settled": g["settled"].mean(), "pending24_n": g["pending24"].sum(),
        "share_current_at_snapshot": g["loan_status"].apply(lambda s: (s == "Current").mean()),
    })
    for k in labs:
        t[k] = g[k].mean()
    t["stop24_excl_settled"] = df[df["settled"] == 0].groupby("issue_q")["stop24"].mean()
    return t.reset_index()


def sensitivity(df, labs):
    rows = []
    for name, (a, b) in {"학습 2010~2012": TRAIN, "평가 2013Q1~2016Q2": EVAL}.items():
        sub = df[(df["issue_d"] >= a) & (df["issue_d"] <= b)]
        sub_q = sub.groupby("issue_q")
        base_q = sub_q["stop24"].mean()
        for k in labs + ["stop24_excl_settled"]:
            s = sub if k != "stop24_excl_settled" else sub[sub["settled"] == 0]
            col = "stop24" if k == "stop24_excl_settled" else k
            q = s.groupby("issue_q")[col].mean()
            rows.append({
                "period": name, "label": k, "n": len(s), "events": int(s[col].sum()),
                "rate": s[col].mean(),
                "diff_vs_base_pp": (s[col].mean() - sub["stop24"].mean()) * 100,
                "loan_disagree_pct": (s[col] != s["stop24"]).mean() * 100,
                "quarter_rate_corr": np.corrcoef(q, base_q.loc[q.index])[0, 1],
                "auc_subgrade": roc_auc_score(s[col], s["grade_num"]),
            })
    return pd.DataFrame(rows)


def final36(df):
    """36개월 대출 중 만기 + 지연이 스냅샷 이전에 끝난 코호트: 최종 상각 대비 24개월 정답"""
    last_ok = SNAP - pd.DateOffset(months=36 + LAG + REPORT_LAG)
    s = df[(df["term_m"] == 36) & (df["issue_d"] >= "2010-01-01") & (df["issue_d"] <= last_ok)]
    g = s.groupby("issue_q")
    t = pd.DataFrame({"n": g.size(), "final_co_rate": g["final_co"].mean(),
                      "stop24_rate": g["stop24"].mean(),
                      "still_open_share": g["loan_status"].apply(
                          lambda x: (~x.isin(["Charged Off", "Fully Paid"])).mean())})
    t["capture_pct"] = t["stop24_rate"] / t["final_co_rate"] * 100
    summ = {"last_issue_month": str(last_ok.date()), "n": int(len(s)),
            "final_co_rate": float(s["final_co"].mean()), "stop24_rate": float(s["stop24"].mean()),
            "capture_pct": float(s["stop24"].sum() / s["final_co"].sum() * 100),
            "quarter_rate_corr": float(np.corrcoef(t["final_co_rate"], t["stop24_rate"])[0, 1]),
            "auc_subgrade_final": float(roc_auc_score(s["final_co"], s["grade_num"])),
            "auc_subgrade_stop24": float(roc_auc_score(s["stop24"], s["grade_num"]))}
    return t.reset_index(), summ


def feature_availability(df):
    feats = [c for c in ORIG_NUMERIC] + ORIG_CATEGORICAL
    yr = df["issue_d"].dt.year
    miss = df[feats].isna().groupby(yr).mean().T
    miss.columns = [str(c) for c in miss.columns]
    first = {}
    for f in feats:
        ok = miss.loc[f][miss.loc[f] < 0.10]
        first[f] = int(ok.index[0]) if len(ok) else None
    miss["first_year_missing_lt10"] = pd.Series(first)
    miss["train_2010_2012_missing"] = df.loc[(yr >= 2010) & (yr <= 2012), feats].isna().mean()
    miss["eval_2013_2018_missing"] = df.loc[yr >= 2013, feats].isna().mean()
    return miss


def policy_drift(df):
    yr = df["issue_d"].dt.year
    g = df.groupby(yr)
    t = pd.DataFrame({
        "n": g.size(), "fico_min": g["fico"].min(), "fico_p05": g["fico"].quantile(.05),
        "loan_amnt_max": g["loan_amnt"].max(), "dti_p99": g["dti"].quantile(.99),
        "share_60m": g["term_m"].apply(lambda s: (s == 60).mean()),
        "share_joint": g["application_type"].apply(lambda s: (s == "Joint App").mean()),
        "share_not_verified": g["verification_status"].apply(lambda s: (s == "Not Verified").mean()),
        "share_grade_AB": g["grade"].apply(lambda s: s.isin(["A", "B"]).mean()),
        "share_grade_EFG": g["grade"].apply(lambda s: s.isin(["E", "F", "G"]).mean()),
    })
    for gr in ["A", "C", "E"]:
        t[f"int_rate_med_{gr}"] = df[df["grade"] == gr].groupby(yr)["int_rate"].median()
    return t.reset_index().rename(columns={"issue_d": "year"})


def lag_evidence():
    full = pd.read_parquet(BASE_PARQUET, columns=["loan_status", "last_pymnt_d",
                                                  "debt_settlement_flag", "policy_flag"])
    since = month_diff(pd.Series(SNAP, index=full.index), full["last_pymnt_d"])
    late = full["loan_status"].eq("Late (31-120 days)")
    co = full["loan_status"].eq("Charged Off")
    st = full["debt_settlement_flag"].eq("Y")
    return {
        "late31_120_n": int(late.sum()),
        "late31_120_months_since_last_pay": since[late].value_counts().sort_index().to_dict(),
        "co_no_settle_n": int((co & ~st).sum()),
        "co_no_settle_le4m_pct": float((since[co & ~st] <= 4).mean() * 100),
        "co_settle_n": int((co & st).sum()),
        "co_settle_le4m_pct": float((since[co & st] <= 4).mean() * 100),
        "note": "스냅샷 시점 31~120일 연체는 마지막 납부 후 0~4개월에 몰려 있고, "
                "채무조정 없는 상각은 마지막 납부 후 4개월 이하가 드묾 -> 상환 중단 후 약 5개월째 상각 기록으로 근사",
    }


def windows(ct):
    pend = ct.set_index("issue_q")["pending24_n"]
    complete = [q for q in pend.index if pend.loc[:q].sum() == 0]
    last_complete_q = complete[-1]
    asof_qs = pd.period_range("2015Q4", "2018Q4", freq="Q")
    replay = []
    for q in asof_qs:
        m = q.asfreq("M", "end") - (HORIZON + LAG + REPORT_LAG)   # 정답 확정된 마지막 실행월
        last_lab = m.asfreq("Q")
        if m != last_lab.asfreq("M", "end"):                       # 분기 전체가 확정된 분기만
            last_lab = last_lab - 1
        replay.append({"asof_q": str(q), "lead_check_issue_q": str(q),
                       "lag_check_latest_labeled_issue_q": str(last_lab)})
    return {
        "label": f"실행 후 {HORIZON}개월 안 상환 중단 후 상각 (STOP 방식)",
        "label_known_after_months": HORIZON + LAG + REPORT_LAG,
        "train_cohort": "2010Q1~2012Q4 (적합 2010Q1~2012Q2, 선택용 2012Q3~Q4)",
        "model_available_from": "2015Q3 (2012Q4 코호트 정답 확정 2015-06)",
        "replay_asof": f"{asof_qs[0]}~{asof_qs[-1]} ({len(asof_qs)}분기)",
        "last_label_complete_issue_q": last_complete_q,
        "replay": replay,
    }


def plot(ct, path):
    t = ct[(ct["issue_q"] >= "2009Q1") & (ct["issue_q"] <= "2018Q4")].copy()
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    ax.axvspan(t.index.get_indexer_for(t[t.issue_q == "2010Q1"].index)[0] - .5,
               t.index.get_indexer_for(t[t.issue_q == "2012Q4"].index)[0] + .5,
               color="#e8e7e2", zorder=0)
    series = [("stop24", "기본: 24개월 내 상환 중단 후 상각", "#2a78d6", "-"),
              ("co24_L5", "비교: 근사 상각월(+5개월) 24개월 내", "#eb6834", "-"),
              ("stop24_settle_adj", "비교: 채무조정 상각 중단월 보정", "#1baf7a", "--")]
    lim = t.index.get_indexer_for(t[t.issue_q == "2016Q3"].index)[0]
    for k, lab, c, ls in series:
        y = t[k].values * 100
        ax.plot(x[:lim + 1], y[:lim + 1], color=c, lw=2, ls=ls, label=lab)
        ax.plot(x[lim:], y[lim:], color=c, lw=1, ls=":", alpha=.6)
    ax.axvline(lim + .5, color="#52514e", lw=1)
    ax.text(lim + .7, ax.get_ylim()[1] * .95, "이후 실행분은\n정답 미확정", fontsize=8,
            color="#52514e", va="top")
    ax.text(x[t.issue_q.tolist().index("2010Q1")], ax.get_ylim()[1] * .95, "학습 코호트",
            fontsize=8, color="#52514e", va="top")
    ax.set_xticks(x[::4]); ax.set_xticklabels(t["issue_q"].values[::4], fontsize=8)
    ax.set_ylabel("상각률 (%)"); ax.grid(axis="y", color="#e8e7e2", lw=.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("실행 분기별 24개월 상각률: 정답 정의에 따른 차이", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    df, n_all = load()
    df, labs = add_labels(df)
    ct = cohort_table(df, labs)
    ct.to_csv(RESULTS / "m02_cohort_labels.csv", index=False, encoding="utf-8-sig")
    sens = sensitivity(df, labs)
    sens.to_csv(RESULTS / "m02_sensitivity.csv", index=False, encoding="utf-8-sig")
    f36, f36s = final36(df)
    f36.to_csv(RESULTS / "m02_final36.csv", index=False, encoding="utf-8-sig")
    fa = feature_availability(df)
    fa.to_csv(RESULTS / "m02_feature_avail.csv", encoding="utf-8-sig")
    pdft = policy_drift(df)
    pdft.to_csv(RESULTS / "m02_policy_drift.csv", index=False, encoding="utf-8-sig")
    lag = lag_evidence()
    win = windows(ct)
    plot(ct, RESULTS / "m02_label_rates.png")
    summary = {"rows_total": n_all, "rows_policy_ok": int(len(df)),
               "rows_policy_excluded": int(n_all - len(df)),
               "final36": f36s, "lag": lag, "windows": win}
    (RESULTS / "m02_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(sens.round(4).to_string())
    print(json.dumps({k: v for k, v in summary.items() if k != "windows"}, ensure_ascii=False,
                     indent=1, default=str))
    print(json.dumps(win, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
