"""m07. 자동화 안전성 점검 (시뮬레이션 표기)

1. 재실행 중복 방지: 같은 분기를 두 번 실행 -> 두 번째는 재생성 없이 기존 결과 반환
2. 오류·변화 주입: 열 누락 / 형식 변경 / 신규 데이터 지연 / 결과 코호트 표본 부족 / FICO 하락
   -> 조용히 잘못된 보고서를 만들지 않고 실패·판정 유보로 기록되는지, 변화는 탐지되는지
산출: results/m07_robustness.csv

실행: .venv/Scripts/python modelops/analysis/m07_robustness.py --quarter 2016Q3
"""
import argparse

import pandas as pd

import inject
from config import RESULTS
from pipeline import Context, run_quarter

EXPECT = {"missing_column": "판정 유보", "type_change": "판정 유보", "delayed_feed": "선행 점검만 유보, 결과 점검은 수행",
          "small_cohort": "판정 유보", "fico_drop": "주의 이상 (선행 탐지)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarter", default="2016Q3")    # 실데이터에서 '주의'인 분기 (후보 실험 없음)
    a = ap.parse_args()
    ctx = Context.load()
    rows = []
    first = run_quarter(ctx, a.quarter)
    second = run_quarter(ctx, a.quarter)
    rows.append({"test": "재실행 중복 방지", "expected": "두 번째 실행은 재생성 안 함",
                 "result": f"1회차 skipped={first.get('skipped', False)}, 2회차 skipped={second.get('skipped', False)}",
                 "pass": bool(second.get("skipped"))})
    for fn in inject.ALL:
        r = run_quarter(ctx, a.quarter, inject=fn, force=True, with_retrain=False)
        res = f"{r['run_status']} / {r['status']} / {r['recommendation']} / {','.join(r.get('reasons', []))}"
        if r.get("error"):
            res += f" / {r['error'][:80]}"
        ok = {"missing_column": r["status"] == "판정 유보",
              "type_change": r["status"] == "판정 유보",
              "delayed_feed": r["run_status"] == "성공" and "lead_new_data_missing" in r.get("reasons", []),
              "small_cohort": r["status"] == "판정 유보",
              "fico_drop": any(x.startswith("lead_") for x in r.get("reasons", []))}[fn.__name__]
        rows.append({"test": f"주입: {fn.__name__}", "expected": EXPECT[fn.__name__], "result": res,
                     "pass": ok})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "m07_robustness.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_colwidth", 140)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
