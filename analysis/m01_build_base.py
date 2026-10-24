"""m01. 원천 CSV -> 대출 단위 기준 테이블(Parquet)

- 원천: Lending Club 실행 대출 2007~2018Q4 (151열). 자유 텍스트(url, desc, title, emp_title) 제외
- 'Does not meet the credit policy' 표기 대출은 policy_flag=0 으로 남기고 분석에서 제외
- 날짜(월) 변환, 실행 시점 파생 변수(term_m, emp_years, grade_num, fico, credit_hist_m)만 생성
- 결과: modelops/data/loans_base.parquet + results/m01_ingest.json (행 수, 스냅샷 월, 원천 해시)

실행: .venv/Scripts/python modelops/analysis/m01_build_base.py
"""
import hashlib
import json
import time

import duckdb
import numpy as np
import pandas as pd

from config import BASE_PARQUET, RAW_CSV, RESULTS

DROP_TEXT = ["url", "desc", "title", "emp_title", "member_id", "zip_code"]
DATE_COLS = ["issue_d", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d",
             "earliest_cr_line", "debt_settlement_flag_date", "settlement_date",
             "hardship_start_date", "hardship_end_date", "payment_plan_start_date"]


def file_sha1(path, block=1 << 24):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    df = con.execute(
        f"SELECT * EXCLUDE ({', '.join('\"' + c + '\"' for c in DROP_TEXT)}) "
        f"FROM read_csv('{RAW_CSV.as_posix()}', all_varchar=true, header=true)"
    ).df()
    n_raw = len(df)
    # 요약 행(빈 id, 'Total amount funded...') 제거
    df = df[df["issue_d"].notna() & df["loan_amnt"].notna()].copy()

    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], format="%b-%Y", errors="coerce")
    non_numeric = set(DATE_COLS) | {
        "id", "term", "grade", "sub_grade", "emp_length", "home_ownership",
        "verification_status", "loan_status", "purpose", "addr_state",
        "initial_list_status", "application_type", "verification_status_joint",
        "pymnt_plan", "hardship_flag", "hardship_type", "hardship_reason",
        "hardship_status", "hardship_loan_status", "disbursement_method",
        "debt_settlement_flag", "settlement_status", "policy_code"}
    for c in df.columns:
        if c not in non_numeric:
            df[c] = pd.to_numeric(df[c].str.rstrip("%"), errors="coerce")

    st = df["loan_status"].fillna("")
    df["policy_flag"] = (~st.str.startswith("Does not meet")).astype(int)
    df["loan_status"] = st.str.replace("Does not meet the credit policy. Status:", "",
                                       regex=False)
    df["term_m"] = df["term"].str.extract(r"(\d+)")[0].astype(float)
    df["emp_years"] = (df["emp_length"].str.replace("< 1", "0", regex=False)
                       .str.extract(r"(\d+)")[0].astype(float))
    df["grade_num"] = df["sub_grade"].map(
        {f"{g}{i}": k * 5 + i for k, g in enumerate("ABCDEFG") for i in range(1, 6)})
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["last_fico"] = (df["last_fico_range_low"] + df["last_fico_range_high"]) / 2
    # 신용 이력 길이 = 실행월 - 최초 신용 개설월 (둘 다 실행 시점 값)
    df["credit_hist_m"] = ((df["issue_d"].dt.year - df["earliest_cr_line"].dt.year) * 12
                           + df["issue_d"].dt.month - df["earliest_cr_line"].dt.month)
    df["issue_q"] = df["issue_d"].dt.to_period("Q").astype(str)

    df.to_parquet(BASE_PARQUET, index=False)
    snapshot = df["last_pymnt_d"].max()
    info = {
        "source": RAW_CSV.name, "source_sha1": file_sha1(RAW_CSV),
        "rows_raw": int(n_raw), "rows_loans": int(len(df)),
        "rows_policy_ok": int(df["policy_flag"].sum()),
        "issue_min": str(df["issue_d"].min().date()), "issue_max": str(df["issue_d"].max().date()),
        "snapshot_last_pymnt_max": str(snapshot.date()),
        "next_pymnt_max": str(df["next_pymnt_d"].max().date()),
        "loan_status_counts": df["loan_status"].value_counts().to_dict(),
        "seconds": round(time.time() - t0, 1),
    }
    (RESULTS / "m01_ingest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
