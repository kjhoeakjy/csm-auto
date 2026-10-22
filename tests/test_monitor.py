"""모니터링 함수 점검: PSI, 스키마 오류 처리, 조기 결과 확정 시점"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from metrics import evaluate, psi  # noqa: E402
from monitor import schema_check  # noqa: E402
from m04_monitor_history import latest_labeled_q  # noqa: E402


def test_psi_zero_for_same_distribution_and_counts_missing():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    assert psi(x, x)[0] < 1e-9
    y = x.copy()
    y[:4000] = np.nan                        # 결측 20% 증가는 PSI 로 잡혀야 함
    assert psi(x, y)[0] > 0.1


def test_schema_check_flags_missing_and_non_numeric_columns():
    bad = pd.DataFrame({"fico": ["700"] * 3})
    r = schema_check(bad)
    assert not r["ok"] and r["errors"]


def test_calibration_z_near_zero_when_calibrated():
    rng = np.random.default_rng(1)
    p = rng.uniform(.02, .3, 200000)
    y = rng.uniform(size=p.size) < p
    r = evaluate(y, p)
    assert abs(r["calib_z"]) < 3 and abs(r["gap_pp"]) < .3


def test_early_and_full_label_availability():
    t = pd.Period("2016Q4", freq="Q")
    assert str(latest_labeled_q(t, 24)) == "2014Q2"    # 30개월 전
    assert str(latest_labeled_q(t, 12)) == "2015Q2"    # 18개월 전
