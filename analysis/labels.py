"""정답(목표변수) 정의와 '언제 정답을 알 수 있었는가' 계산

원천에는 상각일이 없음. 관측되는 것은 마지막 납부월(last_pymnt_d)과 최종 상태뿐.
  stop_d  : 상환 중단 월 = 마지막 납부월 (한 번도 내지 않았으면 실행월)
  stop_m  : 실행 후 상환 중단까지 개월 수
기본 정답(STOP 방식): 실행 후 h개월 안에 상환이 중단되고 결국 상각된 대출 = 1
비교 정답(CO 방식)  : 근사 상각월(stop_d + 지연 L개월)이 실행 후 h개월 안인 상각 = 1
정답 확정 시점      : 실행월 + h + L + 보고 지연 1개월
  (h개월째에 중단한 대출이 상각으로 기록되기까지 L개월이 걸리므로)
"""
import numpy as np
import pandas as pd

HORIZON = 24
LAG = 5          # 상환 중단 -> 상각 기록까지 개월 (m02 에서 4/5/6 민감도)
REPORT_LAG = 1


def month_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.dt.year - b.dt.year) * 12 + (a.dt.month - b.dt.month)


def add_stop(df: pd.DataFrame, settlement: str = "raw", lag: int = LAG) -> pd.DataFrame:
    """stop_d, stop_m 추가.
    settlement='raw'    : 마지막 납부월 그대로
    settlement='adjust' : 채무조정 상각은 합의 상환액이 마지막 납부월에 섞이므로
                          min(마지막 납부월, 채무조정 표시월 - lag) 로 보정
    """
    stop = df["last_pymnt_d"].fillna(df["issue_d"])
    if settlement == "adjust":
        flag_d = df["debt_settlement_flag_date"] - pd.DateOffset(months=lag)
        settled = (df["debt_settlement_flag"] == "Y") & flag_d.notna()
        stop = stop.mask(settled & (flag_d < stop), flag_d)
        stop = stop.mask(stop < df["issue_d"], df["issue_d"])
    df["stop_d"] = stop
    df["stop_m"] = month_diff(stop, df["issue_d"])
    return df


def make_label(df: pd.DataFrame, horizon: int = HORIZON, lag: int = LAG,
               method: str = "stop", settlement: str = "raw") -> pd.Series:
    """정답 y (0/1). method='stop' | 'co'"""
    d = add_stop(df.copy(), settlement=settlement, lag=lag)
    co = d["loan_status"].eq("Charged Off")
    if method == "stop":
        y = co & (d["stop_m"] <= horizon)
    elif method == "co":
        y = co & (d["stop_m"] + lag <= horizon)
    else:
        raise ValueError(method)
    return y.astype(int)


def label_known_date(issue_d: pd.Series, horizon: int = HORIZON, lag: int = LAG) -> pd.Series:
    """이 대출의 h개월 정답이 확정되어 운영자가 알 수 있게 되는 월"""
    return issue_d + pd.DateOffset(months=horizon + lag + REPORT_LAG)


def last_labeled_issue_month(asof: pd.Timestamp, horizon: int = HORIZON,
                             lag: int = LAG) -> pd.Timestamp:
    """시점 asof 에 정답이 확정된 가장 늦은 실행월"""
    return (asof - pd.DateOffset(months=horizon + lag + REPORT_LAG)).to_period("M").to_timestamp()
