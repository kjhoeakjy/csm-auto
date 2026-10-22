"""입력 v2: v1(실행 시점 변수) + 파생 변수 + 지역·직업 정보. 모두 실행 시점에 관측 가능한 값만

추가 변수
  pti           월 상환액 × 12 / 연소득 (상환 부담)
  rbi           리볼빙 잔액 / 연소득
  open_ratio    보유 계좌 / 전체 계좌
  rate_resid    금리 − 같은 실행 분기·같은 세부등급 금리 중앙값 (가격 정책 대비 개별 차이)
  emp_missing   직업명 미기재
  emp_*         직업명 상위 범주 (관리직·간호·교사·운전 등) 표시
  list_whole    전액 대출 채널(initial_list_status = w)
  state_code    거주 주 (LightGBM 범주형)
  zip3_te       우편번호 앞 3자리의 상각률 (학습 구간에서만 계산, 표본 수로 축소)
"""
import numpy as np
import pandas as pd

import features as F
from config import DATA, LEAKAGE

EMP_GROUPS = {"manager": r"manag|director|supervis", "nurse": r"\bnurse|\brn\b", "teacher": r"teach|professor",
              "driver": r"driver|truck", "owner": r"owner|self", "sales": r"sales", "engineer": r"engineer",
              "admin": r"admin|assistant|clerk", "military": r"army|navy|military|air force|usmc"}
STATES = None
V2_NUM = ["pti", "rbi", "open_ratio", "rate_resid", "emp_missing", "list_whole", "zip3_te"] + \
         [f"emp_{k}" for k in EMP_GROUPS]


def attach_text(df: pd.DataFrame) -> pd.DataFrame:
    if "emp_title" in df:
        return df
    t = pd.read_parquet(DATA / "extra_text.parquet")
    return df.merge(t, on="id", how="left")


class Zip3Encoder:
    """우편번호 앞 3자리별 상각률 (베이지안 축소). 적합 구간 정답만 사용"""
    def __init__(self, k=200):
        self.k = k

    def fit(self, zip_code, y):
        z = pd.Series(zip_code).str[:3].values
        d = pd.DataFrame({"z": z, "y": np.asarray(y)})
        self.prior_ = d["y"].mean()
        g = d.groupby("z")["y"].agg(["sum", "count"])
        self.map_ = ((g["sum"] + self.k * self.prior_) / (g["count"] + self.k)).to_dict()
        return self

    def transform(self, zip_code):
        z = pd.Series(zip_code).str[:3]
        return z.map(self.map_).fillna(self.prior_).values


def rate_residual(df):
    med = df.groupby([df["issue_d"].dt.to_period("Q"), "sub_grade"])["int_rate"].transform("median")
    return df["int_rate"] - med


def build(df: pd.DataFrame, zip_enc: Zip3Encoder | None, extended=False) -> pd.DataFrame:
    X = F.build(df, extended=extended)
    inc = df["annual_inc"].replace(0, np.nan)
    X["pti"] = (df["installment"] * 12 / inc).clip(upper=1.5)
    X["rbi"] = (df["revol_bal"] / inc).clip(upper=5)
    X["open_ratio"] = (df["open_acc"] / df["total_acc"].replace(0, np.nan)).clip(0, 1)
    X["rate_resid"] = rate_residual(df)
    et = df["emp_title"].fillna("").str.lower()
    X["emp_missing"] = (et == "").astype(float)
    for k, pat in EMP_GROUPS.items():
        X[f"emp_{k}"] = et.str.contains(pat, regex=True).astype(float)
    X["list_whole"] = (df["initial_list_status"] == "w").astype(float)
    X["zip3_te"] = zip_enc.transform(df["zip_code"]) if zip_enc is not None else np.nan
    X["state_code"] = pd.Categorical(df["addr_state"], categories=sorted(STATE_LIST)).codes
    assert not set(X.columns) & set(LEAKAGE)
    return X


STATE_LIST = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN",
              "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ",
              "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA",
              "WI", "WV", "WY"]
