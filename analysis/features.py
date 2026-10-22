"""모형 입력 생성: 모든 분기에 같은 규칙을 적용 (학습 구간 통계에 의존하는 변환은 모형 파이프라인 안에서)

CORE     : 2007~2009년부터 결측 없이 제공되는 실행 시점 변수 (기준모형·챔피언)
EXTENDED : 2013년부터 완비되는 신용조회 확장 변수 (2013년 이후 코호트로 학습하는 재학습 후보 전용)
"""
import numpy as np
import pandas as pd

from config import LEAKAGE

CORE_NUM = [
    "loan_amnt", "term_m", "int_rate", "installment", "grade_num", "emp_years",
    "log_annual_inc", "dti", "delinq_2yrs", "fico", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "log_revol_bal", "revol_util", "total_acc", "credit_hist_m",
    "collections_12_mths_ex_med", "acc_now_delinq", "chargeoff_within_12_mths",
    "delinq_amnt", "pub_rec_bankruptcies", "tax_liens", "loan_to_income",
]
EXT_NUM = [
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths",
    "avg_cur_bal", "bc_open_to_buy", "bc_util", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd",
    "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m",
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "tot_hi_cred_lim", "total_bal_ex_mort",
    "total_bc_limit", "total_il_high_credit_limit",
]
CAT = {
    "home_ownership": ["MORTGAGE", "RENT", "OWN"],
    "verification_status": ["Not Verified", "Verified", "Source Verified"],
    "purpose": ["debt_consolidation", "credit_card", "home_improvement", "other",
                "major_purchase", "small_business", "car", "medical"],
}
CAT_COLS = [f"{c}_{v}" for c, levels in CAT.items() for v in levels]

# 원천에서 읽어야 하는 열
RAW_COLS = sorted(set(
    [c for c in CORE_NUM if not c.startswith(("log_", "loan_to"))]
    + ["annual_inc", "revol_bal"] + EXT_NUM + list(CAT)
    + ["id", "issue_d", "issue_q", "policy_flag", "grade", "sub_grade",
       "application_type", "initial_list_status", "addr_state"]))


def build(df: pd.DataFrame, extended: bool = False) -> pd.DataFrame:
    """실행 시점 변수만으로 입력 행렬 생성 (행 순서 유지)"""
    X = pd.DataFrame(index=df.index)
    X["log_annual_inc"] = np.log1p(df["annual_inc"].clip(lower=0))
    X["log_revol_bal"] = np.log1p(df["revol_bal"].clip(lower=0))
    X["loan_to_income"] = (df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)).clip(upper=5)
    for c in CORE_NUM:
        if c not in X:
            X[c] = df[c].astype(float)
    # DTI 음수·999 같은 기록 오류는 결측 처리
    X["dti"] = X["dti"].where((X["dti"] >= 0) & (X["dti"] < 100))
    if extended:
        for c in EXT_NUM:
            X[c] = df[c].astype(float)
    for c, levels in CAT.items():
        for v in levels:
            X[f"{c}_{v}"] = (df[c] == v).astype(float)
    assert not set(X.columns) & set(LEAKAGE), "누수 변수가 입력에 포함됨"
    return X


def num_cols(extended: bool = False):
    return CORE_NUM + (EXT_NUM if extended else [])
