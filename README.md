# CSM Auto: 대출 부실 예측모형 자동 점검 시스템

분기별 부실 예측모형 점검 및 수정안 보고 시스템. 예측 부실률과 실제 부실률을 비교하고, 오차 확대 시 수정안을 검증해 담당자에게 보고. 모형 교체 여부는 담당자 결정.

- 결과 화면: https://kjhoeakjy.github.io/csm-auto/ (점검 시점 선택, 권고안 적용 결과 확인)
- 상세 분석 화면: https://kjhoeakjy.github.io/csm-auto/app/detail.html
- 자료: Lending Club 공개 대출 226만 건 (2007~2018). 2015년 4분기~2018년 4분기 13개 분기 재현
- 범위: 공개 자료 기반 프로토타입. 실제 금융회사 운영 사례 아님

## 1. 대상과 정의
| 항목 | 내용 |
|---|---|
| 관리 대상 | 신규 대출의 부실 확률 예측모형 1개 (고객별 모형 아님) |
| 부실 | 대출 실행 후 24개월 내 상환 중단 후 상각. 상각 시점은 마지막 납부월 기준 근사 |
| 실적 확정 | 24개월 부실은 실행 후 30개월, 12개월 부실(중간 신호)은 실행 후 18개월 |
| 판단 원칙 | 각 분기 판단에 해당 시점까지 확정된 정보만 사용. 이후 실적은 사후 평가 전용 |

## 2. 작동 방식
1. 분기별 점검: 신규 대출 구성, 12개월 중간 신호, 24개월 확정 실적의 예측 대비 차이
2. 상태 판정: 정상 / 주의 / 조치 필요 / 판단 보류 (사전 고정 기준, `analysis/rules.py`)
3. 원인 유형 진단: 자료 오류 / 고객 구성 변화 / 확률 수준 오차 / 순위 판별력 저하
4. 수정안 검증 (조치 필요 시만): 현행 유지, 예측 확률 보정 7종, 재학습 4종을 1년 전 자료로 수립 후 이후 실적으로 평가
5. 보고: 상태, 권고, 근거 3개. 보고서 숫자는 계산 결과와 자동 대조, 불일치 시 발행 보류

## 3. 주요 성과 (실자료, 규칙 r4)
| 항목 | 결과 |
|---|---|
| 이상 감지 시점 | 12개월 중간 신호로 6개월 단축 (2016년 4분기 감지, 24개월 실적 기준 2017년 2분기) |
| 2018년 1분기 권고안 사후 확인 | 당시 보유 대출 461,365건 예측 오차 33%에서 13%로, AUC 0.688에서 0.720으로 |
| 운영 방식 비교 (3년) | 고정 모형 27.5%, 분기별 재학습 19.4% (교체 13회), 본 시스템 17.3% (교체 4회) |
| 판정 기준 모의 검증 | 오차 없음 상태의 오경보 0%, 오차 20% 이상 감지율 100% (주입 실험 각 300회) |
| 예측모형 판별력 | AUC 0.68 (2010~2012년 학습, 2013~2016년 평가), 재학습 시 0.72. 공개 연구 동일 조건 0.65~0.71 |

- 오차: |실제 부실률 ÷ 예측 부실률 − 1|
- 판별력(AUC·log loss)은 분기별 재학습 방식이 소폭 우위 (AUC 0.701 대 0.693)

## 4. 폴더
| 경로 | 내용 |
|---|---|
| `index.html`, `app/index.html` | 결과 화면 (`app/simple.html` 원본) |
| `app/detail.html` | 상세 분석 화면 (`app/template.html` 원본) |
| `analysis/` | 자료 정리, 모형, 점검, 판정, 수정안 검증, 재현, 사후 평가, 화면 생성 코드 |
| `runs/<규칙>/<분기>/` | 분기별 판단 근거(`decision.json`), 수정안 비교표 |
| `results/` | 결과 표·그림·요약 |
| `docs/` | 단계별 검증 문서 ([목차](docs/README.md)) |
| `ppt/` | 발표 자료 생성 코드와 결과물 |
| `tests/` | 누수 방지, 실적 확정 시점, 점검 함수, 수정안 엔진, 보고서 대조 테스트 |

## 5. 주요 코드
| 파일 | 역할 |
|---|---|
| `analysis/m01_build_base.py` · `m02_label_audit.py` | 원천 정리, 부실 정의·확정 시점·변수 가용성·누수 검증 |
| `analysis/m03_baseline_backtest.py` | 기준모형(로지스틱·LightGBM) 학습, 분기별 백테스트 |
| `analysis/pipeline.py` | 분기 처리 함수 (점검 → 판정 → 진단 → 수정안 검증 → 근거 기록) |
| `analysis/rules.py` · `candidates.py` · `monitor.py` | 판정 기준, 수정안 검증, 점검 지표 |
| `analysis/scheduler.py` · `m05_replay.py` | 정기 실행(APScheduler), 13개 분기 무인 재현 |
| `analysis/posthoc.py` | 사후 평가 (선택 로직과 분리, 저장된 수정안만 사용) |
| `analysis/m11_policies.py` | 운영 방식 4종 비교 |
| `analysis/m07_robustness.py` · `m10_injection.py` · `m12_label_timing.py` | 오류 주입 시험, 감지 성능 모의 검증, 확정 시점 가정 민감도 |
| `analysis/m13_model_search.py` | 입력 변수·튜닝 탐색 (개선 없음, 기존 모형 유지) |
| `analysis/llm_report.py` | LLM 보고서 초안 (조회 도구 5종, 수치·판정·표현 자동 검사) |
| `analysis/m14_build_simple.py` · `m08_build_app.py` | 결과 화면, 상세 화면 생성 |

## 6. 재실행
```powershell
# 원천 파일: 상위 폴더 data/raw/accepted_2007_to_2018Q4.csv.gz, Python 3.12
uv pip install -r requirements.txt
python analysis/m01_build_base.py; python analysis/m02_label_audit.py
python analysis/m03_baseline_backtest.py; python analysis/m04_monitor_history.py
python analysis/m05_replay.py                  # 또는 scheduler.py --mode replay
python analysis/posthoc.py --rule r4
python analysis/m11_policies.py; python analysis/m07_robustness.py
python analysis/m10_injection.py; python analysis/m12_label_timing.py
python analysis/m08_build_app.py --rule r4; python analysis/m14_build_simple.py --rule r4
python -m pytest tests -q
```
- LLM 보고서(선택): 환경변수 `ANTHROPIC_API_KEY` 설정 후 `python analysis/llm_report.py --rule r4 --model claude-opus-5-5`
- 난수 SEED 42 고정. 같은 입력·코드·규칙으로 재실행 시 기존 결과 재사용

## 7. 규칙 버전
| 버전 | 변경 | 비고 |
|---|---|---|
| r1 | 최초 판정·수정안 기준 | 2017년 이후 전 분기 변경 보류 |
| r2 | 수정안 검증 방식 변경 (1년 전 자료로 수립, 이후 실적으로 평가) | 사후 평가 전 고정 |
| r2.1 | 신규 대출 단일 변수 급변 감지, 자료 누락 시 판단 보류 | 오류 주입 시험 반영 |
| r3 | 원인 유형 진단, 문제 집단 탐색 | |
| r4 (최종) | 재학습 강화, 추세 반영 보정, 등급별 오차 기준 변경 | r3 사후 평가 확인 후 변경, 이전 결과 보존 |

## 8. 한계
- 원천에 상각일 부재, 마지막 납부월 기준 근사 (지연 4·5·6개월 가정 시 13개 분기 중 11개 판정 일치)
- 단일 스냅샷 자료로 과거 시점 재현. 실행 시점 변수는 신청 당시 값으로 가정
- 판단 이후 신규 대출의 실적은 자료 범위 밖. 사후 평가는 판단 시점 보유 대출 기준
- 실행 대출 기준 위험. 거절 신청자 포함 전체 신청자로 일반화 불가
