"""m16. 운영 방식 비교 보조 집계: 분기 단순 평균 대비 평가 대출 수 가중 평균

m11 결과(results/m11_policy_quarter.csv)만 사용. 분기마다 당시 보유 대출 전체를 평가하므로
같은 대출이 여러 분기에 반복 포함되며, 가중 평균은 중복 제거 값이 아님.
실행: .venv/Scripts/python modelops/analysis/m16_policy_weighted.py
"""
import numpy as np
import pandas as pd

from config import RESULTS


def main():
    d = pd.read_csv(RESULTS / "m11_policy_quarter.csv")
    d["err"] = (d["book_oe"] - 1).abs()
    rows = []
    for p, g in d.groupby("policy"):
        rows.append({"policy": p, "changes": int(g["changes_cum"].iloc[-1]),
                     "mean_abs_oe_err": g["err"].mean(),
                     "weighted_abs_oe_err": np.average(g["err"], weights=g["book_n"]),
                     "mean_auc": g["book_auc"].mean(),
                     "weighted_auc": np.average(g["book_auc"], weights=g["book_n"]),
                     "mean_ll": g["book_ll"].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "m16_policy_weighted.csv", index=False, encoding="utf-8-sig")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
