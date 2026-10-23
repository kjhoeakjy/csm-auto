"""모형 운영 자동화 프로젝트 공통 설정

변수 목록은 '실행 시점에 알 수 있는가'를 기준으로 나눔.
  ORIGINATION : 대출 실행(신청) 시점에 관측되는 값 -> 모형 입력 후보
  OUTCOME     : 실행 이후 상환 경과로 정해지는 값 -> 정답 정의에만 사용
  LEAKAGE     : 모형 입력에 절대 쓰지 않는 값 (tests/test_leakage.py 에서 점검)
"""
import os
from pathlib import Path

# 프로젝트 폴더(저장소 최상위) 기준 경로. 환경변수로 위치 변경 가능
MO = Path(os.environ.get("CSM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
_RAW_NAME = "accepted_2007_to_2018Q4.csv.gz"
RAW_CSV = Path(os.environ["CSM_RAW_CSV"]) if os.environ.get("CSM_RAW_CSV") else next(
    (p for p in (MO / "data" / "raw" / _RAW_NAME, MO.parent / "data" / "raw" / _RAW_NAME) if p.exists()),
    MO / "data" / "raw" / _RAW_NAME)
ROOT = MO.parent
DATA = MO / "data"
RESULTS = MO / "results"
REPORTS = MO / "reports"
for p in (DATA, RESULTS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

BASE_PARQUET = DATA / "loans_base.parquet"
SEED = 42

# 원천 스냅샷 시점: 파일에 기록된 마지막 납부월의 최댓값으로 확인 (m01 에서 기록)
SNAPSHOT = "2019-03-01"

# ---- 실행 시점 변수 (수치) ----
ORIG_NUMERIC = [
    "loan_amnt", "term_m", "int_rate", "installment", "grade_num", "emp_years",
    "annual_inc", "dti", "delinq_2yrs", "fico", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "credit_hist_m",
    "collections_12_mths_ex_med", "mths_since_last_major_derog", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths",
    "avg_cur_bal", "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths",
    "delinq_amnt", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc",
    "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",
    "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats", "num_tl_120dpd_2m",
    "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
    "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
    # 2015-12 이후에만 제공되는 변수 (가용성 점검 대상)
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "mths_since_rcnt_il",
    "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util",
    "inq_fi", "total_cu_tl", "inq_last_12m",
]
ORIG_CATEGORICAL = ["home_ownership", "verification_status", "purpose",
                    "initial_list_status", "application_type", "addr_state"]

# ---- 정답 정의에만 쓰는 값 ----
OUTCOME = ["loan_status", "last_pymnt_d", "total_rec_prncp", "out_prncp",
           "debt_settlement_flag", "debt_settlement_flag_date", "settlement_date",
           "hardship_flag", "hardship_start_date"]

# ---- 모형 입력 금지 (실행 이후 정보 또는 스냅샷 시점 값) ----
LEAKAGE = [
    "loan_status", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "out_prncp", "out_prncp_inv",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low", "last_fico",
    "pymnt_plan", "funded_amnt_inv",
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length", "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount",
    "hardship_last_payment_amount", "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date", "settlement_amount",
    "settlement_percentage", "settlement_term",
    # 정답 파생값
    "stop_d", "stop_m", "y", "y12", "label_known_d",
]
