"""정답 정의·정답 확정 시점·누수 방지 점검

실행: .venv/Scripts/python -m pytest modelops/tests -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from config import LEAKAGE, ORIG_CATEGORICAL, ORIG_NUMERIC  # noqa: E402
from labels import label_known_date, last_labeled_issue_month, make_label  # noqa: E402


def _loans(rows):
    df = pd.DataFrame(rows, columns=["issue_d", "last_pymnt_d", "loan_status",
                                     "debt_settlement_flag", "debt_settlement_flag_date"])
    for c in ["issue_d", "last_pymnt_d", "debt_settlement_flag_date"]:
        df[c] = pd.to_datetime(df[c])
    return df


def test_model_inputs_exclude_leakage():
    assert not (set(ORIG_NUMERIC) | set(ORIG_CATEGORICAL)) & set(LEAKAGE)


def test_last_fico_and_payment_fields_are_leakage():
    for c in ["last_fico", "last_pymnt_d", "recoveries", "total_rec_prncp", "loan_status"]:
        assert c in LEAKAGE


def test_stop_label_boundaries():
    df = _loans([
        ("2012-01-01", "2014-01-01", "Charged Off", "N", None),   # 24개월째 중단 -> 1
        ("2012-01-01", "2014-02-01", "Charged Off", "N", None),   # 25개월째 -> 0
        ("2012-01-01", None, "Charged Off", "N", None),           # 한 번도 안 냄 -> 1
        ("2012-01-01", "2013-01-01", "Fully Paid", "N", None),    # 완납 -> 0
    ])
    assert make_label(df, method="stop").tolist() == [1, 0, 1, 0]
    # 근사 상각월 방식(+5개월): 19개월째 중단까지만 1
    df2 = _loans([("2012-01-01", "2013-08-01", "Charged Off", "N", None),
                  ("2012-01-01", "2013-09-01", "Charged Off", "N", None)])
    assert make_label(df2, method="co", lag=5).tolist() == [1, 0]


def test_settlement_adjust_moves_stop_earlier():
    # 채무조정 상환액 때문에 마지막 납부월이 늦게 찍힌 상각: 표시월-5개월로 당김
    df = _loans([("2012-01-01", "2014-06-01", "Charged Off", "Y", "2014-03-01")])
    assert make_label(df, method="stop", settlement="raw").tolist() == [0]
    assert make_label(df, method="stop", settlement="adjust").tolist() == [1]


def test_label_known_date_and_asof_cutoff():
    issue = pd.Series(pd.to_datetime(["2013-06-01"]))
    assert label_known_date(issue).iloc[0] == pd.Timestamp("2015-12-01")
    assert last_labeled_issue_month(pd.Timestamp("2015-12-01")) == pd.Timestamp("2013-06-01")


@pytest.mark.parametrize("asof", ["2015-12-01", "2017-03-01", "2018-12-01"])
def test_no_label_after_asof(asof):
    asof = pd.Timestamp(asof)
    issue = pd.Series(pd.date_range("2010-01-01", "2018-12-01", freq="MS"))
    usable = issue[issue <= last_labeled_issue_month(asof)]
    assert (label_known_date(usable) <= asof).all()
