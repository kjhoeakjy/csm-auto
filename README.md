# CSM Auto — 신용평가모형 운영 자동화

신규 대출의 **24개월 내 상각 위험을 예측하는 모형 1개**를 매 분기 자동으로 점검하고, 문제가 생기면 **유지·재보정·재학습 중 당시 데이터로 검증된 조치**를 담당자에게 권고하는 시스템. 고객별 점수를 바꾸는 것이 아니라, 여러 대출 고객에게 공통으로 적용되는 위험 예측모형 자체의 예측 성능을 관리함.

- 데이터: Lending Club 실행 대출 2007~2018 (226만 건), 공개 데이터 기반 **과거 분기 재현(as-of replay)**. 실제 금융회사 운영 아님
- 화면: `app/index.html` (단일 파일, 인터넷 불필요). 분기 선택 → **자동 점검 실행** → 상태 1개 · 권고 1줄 · 근거 3개
- 변경 권한: 시스템은 제안·검증·보고까지. 모형 교체는 담당자 승인

## 한 분기 처리 흐름 (`analysis/pipeline.py`)
```
정기 실행 (scheduler.py, APScheduler)  ─ 과거 재현도 같은 함수 호출
  └ as-of 자르기: 그 분기에 알 수 있었던 데이터만
     ├ 선행 점검   신규 대출 스키마·결측·PSI·점수 분포·구성 (정답 불필요)
     ├ 중간 신호   12개월 상각 (실행 후 18개월에 확인) ─ 24개월 정답이 아닌 조기 경고
     └ 후행 검증   24개월 상각 (실행 후 30개월에 확정) ─ 실제/예측, 과산포 보정 z, AUC, 집단별 오차
  → 판정 (rules.py: 정상 / 주의 / 조치 검토 / 판정 유보)
  → 문제 유형 진단: 데이터 오류 · 분포 변화만 · 확률 수준 오차(순위 유지) · 순위 악화
  → [조치 검토일 때만] 개선 후보 10종을 1년 전 시점 데이터로 적합해 이후 확정 코호트로 검증 (모의 배포)
       유지 / 재보정 6종(수준·Platt·beta·isotonic·등급별·조기 결과 반영) / 재학습 3종(로지스틱·LightGBM·확장 변수)
       확률 수준 오차 → 재보정 우선, 순위 악화 → 재학습 우선, 기준 미달 → 변경 보류
  → 문제 집단 탐색 (Benjamini-Hochberg) → 권고 + 근거 3개 → decision.json
  → 보고서 생성 + 숫자·분기 자동 대조 (불일치 시 발행 보류) → 화면
```

## 폴더
| 경로 | 내용 |
|---|---|
| `analysis/m01~m04` | 원천 정리, 정답·관측 시점·누수 검증, 기준모형·백테스트, 선행·중간·후행 점검 |
| `analysis/pipeline.py` `rules.py` `candidates.py` `monitor.py` | 분기 처리 엔진 |
| `analysis/m05_replay.py` `scheduler.py` | 13분기 무인 재현 (2015Q4~2018Q4) |
| `analysis/posthoc.py` | 사후 평가 (선택 로직과 분리, 저장된 후보만 읽음) |
| `analysis/m07_robustness.py` `inject.py` `m10_injection.py` | 오류 주입 시험, 탐지율·오경보·지연 시뮬레이션 |
| `analysis/m11_policies.py` | **운영 정책 4종 비교**: 고정 / 매 분기 재학습 / 단순 임계값 / 제안 시스템 |
| `analysis/m12_label_timing.py` | 결과 확정 시점 가정(상각 지연 4·5·6개월) 민감도 |
| `analysis/report.py` `llm_report.py` | 규칙 기반 보고서, LLM 보고서(도구 조회 + 수치 대조) |
| `analysis/m08_build_app.py` → `app/index.html` | 결과 화면 |
| `docs/` | 단계별 검증 문서, 완료 기준 점검표 |
| `runs/<규칙 버전>/<분기>/decision.json` | 분기별 정량 근거, 후보 비교표 |
| `results/` | 표·그림·요약 |
| `tests/` | 누수·코호트 경계·확정 시점·모니터링·후보 엔진·보고서 대조 테스트 |

## 다시 실행
```powershell
# 원천: data/raw/accepted_2007_to_2018Q4.csv.gz (상위 폴더). Python 3.12
uv pip install -r requirements.txt
python analysis/m01_build_base.py; python analysis/m02_label_audit.py
python analysis/m03_baseline_backtest.py; python analysis/m04_monitor_history.py
python analysis/scheduler.py --mode replay      # 또는 m05_replay.py
python analysis/posthoc.py --rule r4
python analysis/m07_robustness.py; python analysis/m10_injection.py
python analysis/m11_policies.py; python analysis/m12_label_timing.py
python analysis/m08_build_app.py --rule r4
python -m pytest tests -q
```
LLM 보고서(선택): 환경변수 `ANTHROPIC_API_KEY` 설정 후 `python analysis/llm_report.py --rule r3 --model claude-haiku-5-5`

## 지켜야 할 원칙
- 분기 T의 선택은 T까지 확정된 결과만 사용. 미래 코호트는 사후 평가 전용 (`candidates.py`·`posthoc.py` 분리, 테스트로 고정)
- 상각일은 원천에 없음 → 마지막 납부월 기준 근사, 지연 4·5·6개월 민감도 보고
- 규칙 변경은 버전으로 남김 (r1 → r2 → r2.1 → r3 → r4, `rules.py` 상단에 이유 기록)
- 실데이터 결과와 주입 시뮬레이션 결과는 파일·표기 분리
