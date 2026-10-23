"""판단 시점(분기 말일) 정의와 프로젝트 경로 설정 점검"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from config import MO  # noqa: E402
from labels import HORIZON, LAG, REPORT_LAG  # noqa: E402
from pipeline import asof_month  # noqa: E402

QUARTERS = pd.period_range("2015Q4", "2018Q4", freq="Q")


def test_asof_is_quarter_end():
    assert asof_month(pd.Period("2018Q1")) == pd.Timestamp("2018-03-31")
    assert asof_month(pd.Period("2016Q4")) == pd.Timestamp("2016-12-31")


@pytest.mark.parametrize("t", QUARTERS)
def test_quarter_end_cutoff_matches_month_records(t):
    """월 단위 자료(매월 1일 표기): 분기 말일 기준 선택은 분기 마지막 달 자료까지 포함하고 다음 달 자료는 제외"""
    issue = pd.date_range("2012-01-01", "2016-12-01", freq="MS")
    known = issue + pd.DateOffset(months=HORIZON + LAG + REPORT_LAG)
    asof = asof_month(t)
    sel = known[known <= asof]
    last_month = t.asfreq("M", "end").to_timestamp()
    assert sel.max() <= last_month
    assert (known[known > asof] >= last_month + pd.DateOffset(months=1)).all()


def test_project_root_is_repository_folder():
    assert (MO / "analysis" / "config.py").exists()
    assert (MO / "app" / "fonts").exists()
