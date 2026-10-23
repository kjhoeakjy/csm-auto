"""m10. 주입 실험 (시뮬레이션): 판정 규칙의 탐지율·오경보·탐지 지연

실제 코호트의 운영 모형 점수 p 를 그대로 쓰고, 정답만 모의 생성:
  y ~ Bernoulli(sigmoid(logit(p) + log(k) 근사 + u_q)),  u_q = 분기 공통 변동 (개발 구간 과산포 phi 재현)
  k = 1.00 (변화 없음, 오경보 측정) / 1.05 / 1.10 / 1.20 / 1.30
- 탐지율: 두 연속 코호트에 같은 k 주입 시 '조치 검토'(24개월 규칙) 판정 비율
- 지연: k 가 분기마다 0.03 씩 오르는 추세 주입 시 조치 검토까지 걸린 분기 수
결과는 실데이터 결과와 분리해 results/m10_injection.csv 로 저장

실행: .venv/Scripts/python modelops/analysis/m10_injection.py
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from config import DATA, RESULTS, SCORES, SEED
from rules import R

PHI = json.loads((RESULTS / "m04_summary.json").read_text(encoding="utf-8"))["phi_dev"]
COHORTS = ["2013Q2", "2013Q3", "2013Q4", "2014Q1"]          # 운영 재현 초기 실제 코호트 크기·점수 분포
B = 300


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def shift_for_ratio(p, k):
    """평균 예측 대비 k 배가 되는 logit 이동량"""
    return 0.0 if k == 1 else brentq(lambda a: (1 / (1 + np.exp(-(_logit(p) + a)))).mean() - k * p.mean(), -3, 3)


def sigma_u(p):
    """분기 공통 변동의 표준편차: Var(O-E) = phi * Σp(1-p) 가 되도록"""
    n, v = len(p), (p * (1 - p)).sum()
    extra_sd_rate = np.sqrt((PHI - 1) * v) / n
    return extra_sd_rate / (p * (1 - p)).mean()


def stat(y, p):
    o, e = y.sum(), p.sum()
    z = (o - e) / np.sqrt((p * (1 - p)).sum()) / np.sqrt(PHI)
    return o / e - 1, z


def act(s0, s1):
    rel0, z0 = s0
    rel1, z1 = s1
    return (abs(rel0) >= R["lag_rel_act"] and abs(z0) >= R["lag_z_act"]
            and rel0 * rel1 > 0 and abs(z1) >= R["lag_z_persist"])


def warn(s0):
    return abs(s0[0]) >= R["lag_rel_warn"] and abs(s0[1]) >= R["lag_z_warn"]


def simulate(ps, k, rng):
    out = []
    for p in ps:
        a = shift_for_ratio(p, k)
        u = rng.normal(0, sigma_u(p))
        q = 1 / (1 + np.exp(-(_logit(p) + a + u)))
        out.append(stat((rng.uniform(size=len(p)) < q).astype(int), p))
    return out


def main():
    sc = pd.read_parquet(SCORES, columns=["issue_q", "p_lgbm"])
    ps = [sc.loc[sc.issue_q == q, "p_lgbm"].values for q in COHORTS]
    rng = np.random.default_rng(SEED)
    rows = []
    for k in [1.00, 1.05, 1.10, 1.20, 1.30]:
        n_act = n_warn = 0
        for _ in range(B):
            i = rng.integers(0, len(ps) - 1)
            s1, s0 = simulate([ps[i], ps[i + 1]], k, rng)     # 직전, 이번
            n_act += act(s0, s1)
            n_warn += warn(s0) or act(s0, s1)
        rows.append({"test": "고정 차이", "k": k, "act_rate": n_act / B, "warn_or_act_rate": n_warn / B,
                     "n_cohort": int(np.mean([len(p) for p in ps]))})
    # 추세 주입: 1.00 에서 시작해 분기마다 +0.03
    delays = []
    for _ in range(B):
        prev = None
        for t in range(12):
            k = 1 + 0.03 * t
            p = ps[t % len(ps)]
            s = simulate([p], k, rng)[0]
            if prev is not None and act(s, prev):
                delays.append(t)
                break
            prev = s
        else:
            delays.append(np.nan)
    d = np.array(delays, dtype=float)
    rows.append({"test": "추세(+0.03/분기)", "k": None, "act_rate": float(np.mean(~np.isnan(d))),
                 "delay_median_q": float(np.nanmedian(d)), "delay_p90_q": float(np.nanpercentile(d, 90)),
                 "k_at_detection_median": float(1 + 0.03 * np.nanmedian(d))})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "m10_injection.csv", index=False, encoding="utf-8-sig")
    print("phi", PHI)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
