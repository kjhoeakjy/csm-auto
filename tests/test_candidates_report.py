"""후보 엔진 시점 차단·재보정 동작·보고서 수치 대조 점검 (합성 데이터)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from candidates import EarlyIntercept, Intercept, run_candidates  # noqa: E402
from report import verify  # noqa: E402


def test_intercept_matches_mean():
    rng = np.random.default_rng(0)
    p = rng.uniform(.02, .3, 50000)
    y = (rng.uniform(size=p.size) < np.clip(p * 1.3, 0, 1)).astype(int)
    q = Intercept().fit(p, y).predict(p)
    assert abs(q.mean() - y.mean()) < 1e-6


def test_early_intercept_moves_up_when_early_losses_exceed():
    rng = np.random.default_rng(1)
    p = rng.uniform(.02, .3, 50000)
    emap = lambda x: np.asarray(x) * 0.5                      # 12개월 기대 = 24개월의 절반
    y12 = (rng.uniform(size=p.size) < p * 0.5 * 1.3).astype(int)
    m = EarlyIntercept(emap).fit(p, y12)
    assert m.a_ > 0 and m.predict(p).mean() > p.mean()


def test_candidate_engine_rejects_future_labels():
    lab = pd.DataFrame({"q": pd.PeriodIndex(["2014Q1"] * 3, freq="Q"), "y": [0, 1, 0],
                        "p": [.1, .2, .1], "grade": ["A", "B", "C"],
                        "issue_d": pd.to_datetime(["2014-01-01"] * 3),
                        "label_known_d": pd.to_datetime(["2016-08-01"] * 3)})
    lab12 = lab.assign(label12_known_d=pd.to_datetime(["2015-08-01"] * 3), y12=[0, 0, 0])
    with pytest.raises(AssertionError):
        run_candidates(lab, lab12, pd.Timestamp("2016-06-01"), {}, None, with_retrain=False)


def test_report_verify_catches_changed_number():
    dec = {"quarter": "2018Q1", "lag": [{"cohort_q": "2015Q3", "actual": 0.1341, "pred": 0.1056,
                                         "o_e_ratio": 1.2690, "n": 110489}]}
    ok = "2015Q3 실행 110,489건: 실제 13.4%, 예측 10.6%, 실제/예측 1.27"
    assert verify(ok, dec)["verified"]
    assert not verify(ok.replace("13.4%", "15.2%"), dec)["verified"]
    assert not verify(ok.replace("2015Q3", "2016Q1"), dec)["verified"]


def test_report_verify_catches_swapped_fields():
    """숫자는 근거에 있으나 항목이 뒤바뀐 경우(실제와 예측 교환) 불일치로 처리"""
    dec = {"quarter": "2018Q1",
           "lag": [{"cohort_q": "2015Q3", "actual": 0.1341, "pred": 0.1056, "o_e_ratio": 1.2690, "n": 110489}],
           "early": [{"cohort_q": "2016Q3", "actual": 0.0820, "pred": 0.0603, "o_e_ratio": 1.36, "n": 99120}]}
    ok = "2015년 3분기 대출 실제 상각률 13.4%, 예측 10.6%. 2016Q3 실행분 12개월 결과는 기대의 1.36배"
    assert verify(ok, dec)["verified"]
    swapped = "2015년 3분기 대출 실제 상각률 10.6%, 예측 13.4%"
    r = verify(swapped, dec)
    assert not r["verified"] and r["field_mismatch"]
    # 다른 코호트의 값을 이 코호트에 붙인 경우
    assert not verify("2015Q3 실행분 실제/예측 1.36", dec)["verified"]
