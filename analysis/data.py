"""분석용 대출 테이블 적재 (정책 미달 제외, 기본 정답·정답 확정월 부착)"""
import pandas as pd

from config import BASE_PARQUET
from features import RAW_COLS
from labels import label_known_date, make_label

OUTCOME_COLS = ["loan_status", "last_pymnt_d", "debt_settlement_flag",
                "debt_settlement_flag_date"]


def load_loans(start="2010-01-01", end="2018-12-31") -> pd.DataFrame:
    df = pd.read_parquet(BASE_PARQUET, columns=sorted(set(RAW_COLS + OUTCOME_COLS)))
    df = df[(df["policy_flag"] == 1) & (df["issue_d"] >= start) & (df["issue_d"] <= end)]
    df = df.sort_values(["issue_d", "id"]).reset_index(drop=True)
    df["y"] = make_label(df)
    df["y12"] = make_label(df, horizon=12)          # 조기 결과 (정답 확정 18개월)
    df["label_known_d"] = label_known_date(df["issue_d"])
    return df
