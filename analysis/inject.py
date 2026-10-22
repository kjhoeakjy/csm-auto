"""시험용 주입 (시뮬레이션). 실데이터 결과와 섞이지 않도록 decision_<이름>.json 으로 따로 저장

오류 주입: 열 누락, 형식 변경, 신규 데이터 지연(0건), 결과 코호트 표본 부족
변화 주입: 신규 대출 FICO 하락 (선행 점검 탐지 확인용)
"""
import numpy as np


def missing_column(v):
    v = dict(v)
    v["new"] = v["new"].drop(columns=["fico"])
    return v


def type_change(v):
    v = dict(v)
    new = v["new"].copy()
    new["dti"] = new["dti"].astype(str)
    v["new"] = new
    return v


def delayed_feed(v):
    v = dict(v)
    v["new"] = v["new"].iloc[:0]
    return v


def small_cohort(v):
    """최신 확정 코호트가 1%만 도착"""
    v = dict(v)
    lab = v["lab24"]
    last = lab["q"].max()
    keep = (lab["q"] != last) | (np.random.default_rng(0).uniform(size=len(lab)) < .01)
    v["lab24"] = lab[keep]
    return v


def fico_drop(v):
    v = dict(v)
    new = v["new"].copy()
    new["fico"] = new["fico"] - 40
    v["new"] = new
    return v


ALL = [missing_column, type_change, delayed_feed, small_cohort, fico_drop]
