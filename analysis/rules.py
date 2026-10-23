"""판정 규칙과 후보 선택 기준 (사전 고정, 버전 관리)

r1 (2026-10-22 고정, 운영 재현 구간 결과를 보기 전에 확정)
- 유의성: 분기 과산포 phi 를 개발 구간(2010~2012)에서 추정해 z/√phi 사용
- 실질성: 실제/예측 비율이 10% 이상 벗어날 때 '의미 있는 차이'로 봄 (업무 기준, 데이터로 튜닝하지 않음)
- 후보 선택 기준은 시점 T 에 확정된 코호트(선택용 2분기)에서만 계산

r2 (2026-10-22, r1 운영 재현 직후·사후 평가 실행 전 고정. 판정 규칙은 r1 과 동일, 후보 검증만 변경)
- r1 결과: 2017Q1 이후 모든 후보가 '보정 오차 5% 이내' 절대 기준 미달 -> 변경 보류 반복.
  결과 확정 지연(30개월) 동안 수준이 계속 이동해 24개월 결과로 적합한 후보가 구조적으로 뒤처짐
- 변경 1. 모의 배포 검증: T-4분기 시점에 가용했던 데이터만으로 후보 적합 -> 그 뒤 확정된 최근 4분기 코호트로 평가
- 변경 2. 조기 결과 반영 재보정 후보 추가 (12개월 결과, 18개월에 확정)
- 변경 3. 통과 기준: 유지 대비 보정 오차 감소 (절대 5% 기준 대신). 5% 초과 잔여 오차는 권고에 표시

r2.1 (2026-10-22, 주입 시험 결과 반영. 후보 엔진은 r2 와 동일)
- 신규 대출 FICO 일괄 -40 주입을 선행 점검이 놓침 (변수 3개 이상 급변 조건). 단일 변수라도
  직전 4분기 대비 PSI ≥ 0.25 이면 주의 사유로 기록하도록 보완
- 신규 데이터 미도착·결과 코호트 표본 부족 시 예외 대신 판정 유보로 처리

r3 (2026-10-22, 사후 평가 결과와 무관한 설계 보완: 문제 유형별 조치 선택 + 문제 집단 탐색)
- 판정 뒤 문제 유형 진단: 데이터 오류 / 분포 변화만 / 확률 수준 오차(순위 유지) / 순위·확률 모두 악화
- 확률 수준 오차면 재보정 계열 우선. 재학습은 재보정보다 log loss 1%p 이상·AUC 0.01 이상 더 좋을 때만
- 순위 악화면 재학습 계열 우선 (재보정은 순위를 바꾸지 못함)
- 문제 집단: 등급·기간·목적·FICO·DTI 구간별 실제/예측을 전체 수준 대비 검정, Benjamini-Hochberg 5%

r4 (2026-10-22, r3 사후 평가 확인 후. 판정·선택 기준은 r3 와 동일, 후보만 추가·강화)
- 재학습 LightGBM: 적합 창 마지막 2분기로 시간순 조기 종료 (r3 는 챔피언 반복 수 128 고정 → 큰 창에서 과소적합)
- 후보 추가: 추세 반영 재보정(최근 4개 확정 코호트의 수준 차이 추세를 3분기 앞으로 연장), 최근 가중 재학습(반감기 4분기)
- 집단 기준 변경: "어느 등급도 5%p 넘게 악화 금지" → "상각 건수 가중 평균 등급 오차가 유지보다 나빠지지 않고,
  어느 등급도 유지의 최악 등급 오차를 넘지 않음". r3 기준은 전체 수준을 올리는 보정이 이미 반대 방향인 A등급을
  기계적으로 악화시켜 모든 수준 보정을 탈락시킴 (5개 등급 개선, 1개 등급 악화도 탈락)
- 주의: 사후 평가에서 r3 가 보수적이었던 것을 본 뒤의 변경임. 선택 기준은 그대로 두고 후보만 늘렸으며 r3 결과도 보존

r5 (2026-10-23, 탐색 단계. m17·m18·m19 빠른 시험 결과를 보고 결정)
- 운영 모형 v2: 첫 판단 시점에 결과가 확정된 2010~2013Q2 실행분 전체, 신용조회 확장 입력, 최근 가중 (시간외 AUC 0.681 -> 0.701)
- 후보 추가: 조기 결과 추세 반영 재보정. 12개월 결과 확정 최근 4분기의 수준 차이 추세를 보유 대출 중심 시점까지 연장
- 확률 수준 오차의 선택 규칙: 조기 결과 추세 기준 수준 이동량이 logit 0.10(약 10%) 이상이면 해당 재보정 선택, 미만이면 갱신 보류.
  24개월 확정 코호트 모의 배포 검증은 비교표로 유지 (2016년처럼 오래된 코호트와 최근 코호트의 방향이 반대인 구간에서
  오래된 코호트 검증이 최신 신호를 반영한 후보를 구조적으로 탈락시킴)
- 후보 실험 실행 조건: 조치 검토, 또는 주의 중 12개월 조기 결과 기준 초과
- 주의: 같은 기간 결과를 보고 정한 규칙이므로 독립 검증 아님. r4 결과 보존
"""
RULE_VERSION = "r5"

R = {
    # 후행 검증 (24개월, 최신 확정 코호트)
    "lag_rel_act": 0.10, "lag_z_act": 3.0, "lag_z_persist": 2.0,
    "lag_rel_warn": 0.05, "lag_z_warn": 2.0,
    # 조기 결과 점검 (12개월)
    "early_rel": 0.10, "early_z": 3.0,
    # 선행 점검
    "score_psi_warn": 0.10, "feat_psi_recent_warn": 0.25, "n_feat_warn": 1,
    "miss_jump_pp": 5.0, "volume_low": 0.5,
    # 표본 기준
    "min_cohort_n": 5000, "min_new_n": 5000,
    # 후보 선택
    "sel_quarters": 4, "recal_fit_quarters": 2, "retrain_window_quarters": 8,
    "pseudo_origin_quarters": 4,            # 모의 배포 시점 = T - 4분기
    "early_fit_quarters": 2,                # 조기 결과 반영 재보정: 최신 12개월 확정 코호트 2분기
    "min_rel_logloss_gain": 0.002,          # 유지 대비 log loss 0.2% 이상 개선
    "residual_oe_dev": 0.05,                # 통과 후에도 |실제/예측 - 1| > 5% 면 잔여 오차 표시
    # (r1: "max_sel_oe_dev": 0.05 절대 기준, sel_quarters 2, 모의 배포 없음)
    "max_auc_drop": 0.005,
    "max_group_worsen": 0.05,               # 등급별 |실제/예측-1| 이 유지 대비 5%p 넘게 나빠지면 탈락
    "simpler_tolerance": 0.001,             # log loss 차이 0.1% 이내면 단순한 후보 선택
    "bootstrap_b": 200,
    # r3 문제 유형·조치 선택
    "auc_degrade": 0.02,                    # 기준 AUC 대비 0.02 이상 하락이 2코호트 연속이면 순위 악화
    "retrain_extra_ll": 0.01, "retrain_extra_auc": 0.01,
    # r3 문제 집단 탐색
    "seg_min_n": 2000, "seg_min_events": 50, "seg_rel": 0.10, "seg_fdr": 0.05,
    # r5 조기 결과 추세 반영 재보정
    "early_trend_quarters": 4, "early_trend_min_shift": 0.10,
}

PROBLEM = {"data": "데이터 오류", "drift_only": "입력 분포 변화만", "calibration": "확률 수준 오차 (순위 유지)",
           "ranking": "순위·확률 모두 악화", "none": "이상 없음"}


def diagnose(status: str, reasons: list, lag: list, ref_auc: float | None) -> str:
    """판정 결과를 문제 유형으로 분류 (조치 후보군 결정에 사용)"""
    if status == "판정 유보":
        return "data"
    outcome = any(r.startswith(("lag24", "early12")) for r in reasons)
    if not outcome:
        return "drift_only" if reasons else "none"
    aucs = [l.get("auc") for l in lag[:2] if l.get("auc") is not None]
    if ref_auc is not None and len(aucs) == 2 and all(a <= ref_auc - R["auc_degrade"] for a in aucs):
        return "ranking"
    return "calibration"

STATUS = ["정상", "주의", "조치 검토", "판정 유보"]
RECO = {"keep": "현행 유지", "recal": "재보정 검토", "retrain": "재학습 검토",
        "investigate": "조사 (변경 보류)", "hold": "판정 유보"}


def _sig(rel, z, rel_t, z_t):
    return abs(rel) >= rel_t and abs(z) >= z_t


def judge(lead: dict | None, early: list[dict], lag: list[dict]) -> dict:
    """lead: 이번 분기 선행 점검 요약 / early, lag: 최근 2개 판단 결과 [이번, 직전]
    반환: status, reasons(규칙 id 목록), direction(+1 실제>예측)"""
    reasons, status = [], "정상"
    if lead is not None and not lead.get("schema_ok", True):
        return {"status": "판정 유보", "reasons": ["schema_error"], "direction": 0}
    if not lag or lag[0]["n"] < R["min_cohort_n"]:
        return {"status": "판정 유보", "reasons": ["lag_sample_small"], "direction": 0}

    l0 = lag[0]
    l1 = lag[1] if len(lag) > 1 else None
    direction = 1 if l0["rel_err"] > 0 else -1
    if _sig(l0["rel_err"], l0["z_adj"], R["lag_rel_act"], R["lag_z_act"]) and l1 is not None \
            and l1["rel_err"] * l0["rel_err"] > 0 and abs(l1["z_adj"]) >= R["lag_z_persist"]:
        status = "조치 검토"; reasons.append("lag24_material_persistent")
    elif _sig(l0["rel_err"], l0["z_adj"], R["lag_rel_warn"], R["lag_z_warn"]):
        status = "주의"; reasons.append("lag24_drift")

    if early:
        e0 = early[0]
        e1 = early[1] if len(early) > 1 else None
        if _sig(e0["rel_err"], e0["z_adj"], R["early_rel"], R["early_z"]):
            if e1 is not None and _sig(e1["rel_err"], e1["z_adj"], R["early_rel"], R["early_z"]) \
                    and e1["rel_err"] * e0["rel_err"] > 0:
                if status != "조치 검토":
                    direction = 1 if e0["rel_err"] > 0 else -1
                status = "조치 검토"; reasons.append("early12_material_2q")
            else:
                if status == "정상":
                    status = "주의"
                reasons.append("early12_material")

    if lead is not None:
        lead_flags = []
        if not lead.get("skipped") and (lead["score_psi_ref"] >= R["score_psi_warn"]
                                        or lead["score_psi_recent"] >= R["score_psi_warn"]):
            lead_flags.append("score_shift")
        if lead.get("skipped"):
            lead_flags.append("new_data_missing")
        elif lead["n_feat_psi_recent_ge_0.25"] >= R["n_feat_warn"]:
            lead_flags.append("feature_shift")
        if not lead.get("skipped") and lead["miss_max_delta_recent_pp"] >= R["miss_jump_pp"]:
            lead_flags.append("missing_jump")
        if lead.get("volume_vs_recent") is not None and lead["volume_vs_recent"] < R["volume_low"]:
            lead_flags.append("volume_drop")
        if lead["n"] < R["min_new_n"] and not lead.get("skipped"):
            lead_flags.append("new_sample_small")
        if lead_flags and status == "정상":
            status = "주의"
        reasons += [f"lead_{f}" for f in lead_flags]
    return {"status": status, "reasons": reasons, "direction": direction}
