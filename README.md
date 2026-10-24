# CSM Auto (신용평가모형 자동 점검 시스템)

분기별 신용평가모형 점검 및 수정안 보고 시스템. 예측 상각률과 실제 상각률을 비교하고 오차 확대 시 수정안을 검증해 담당자에게 보고. 모형 갱신 여부는 담당자 결정.

- **결과 화면** https://kjhoeakjy.github.io/csm-auto/ (점검 시점 선택, 권고안 적용 결과 확인)
- **상세 분석 화면** https://kjhoeakjy.github.io/csm-auto/app/detail.html
- **자료** Lending Club 공개 대출 226만 건 (2007~2018). 2015년 4분기~2018년 4분기 13개 분기 재현
- **범위** 공개 자료 기반 프로토타입. 실제 금융회사 운영 사례 아님
- **한 장 요약** [docs/08_프로젝트_요약.md](docs/08_프로젝트_요약.md)

## 1. 대상과 정의
| 항목 | 내용 |
|---|---|
| 관리 대상 | 신규 대출 신용평가모형 1개 (고객별 모형 아님) |
| 상각률 | 대출 실행 후 24개월 내 상환 중단 후 손실 처리(상각) 비율. 상각 시점은 마지막 납부월 기준 근사 |
| 실적 확정 | 24개월 상각률은 실행 후 30개월, 12개월 상각률(중간 신호)은 실행 후 18개월 |
| 판단 시점 | 분기 말일 (예시 2018년 1분기 판단은 2018-03-31). 자료는 월 단위이며 분기 마지막 달 실행 대출과 그 달까지 확정된 실적 포함 |
| 판단 원칙 | 각 분기 판단에 해당 시점까지 확정된 정보 사용. 이후 실적은 사후 평가 전용 |

## 2. 작동 방식
1. **분기별 점검** 신규 대출 구성, 12개월 중간 신호, 24개월 확정 실적의 예측 대비 차이
2. **상태 판정** 정상 / 주의 / 조치 필요 / 판단 보류 (사전 고정 기준, `analysis/rules.py`. 코드와 `runs/` 기록의 명칭은 조치 검토, 판정 유보이며 화면과 문서에서 조치 필요, 판단 보류로 표기)
3. **원인 유형 진단** 자료 오류 / 고객 구성 변화 / 확률 수준 오차 / 순위 판별력 저하
4. **수정안 검증 (조치 필요 시)** 현행 유지, 예측 확률 보정 7종, 재학습 4종을 1년 전 자료로 수립 후 이후 실적으로 평가
5. **보고** 상태, 권고, 근거 3개. 규칙 기반 문장과 Claude Opus 5.5 초안. 보고서 숫자는 계산 결과와 자동 대조(숫자 존재 여부와 실제, 예측 등 항목별 값), 불일치 시 발행 보류

## 3. 주요 성과 (실자료, 운영 모형 v2, 규칙 r5)
| 항목 | 결과 |
|---|---|
| 예측모형 변별력 | 시간외 AUC 0.701 (2010~2013년 상반기 학습, 2013년 하반기~2016년 평가). 기존 v1 0.681. 공개 연구 동일 조건 0.65~0.71 |
| 이상 경고 시점 | 1년 중간 실적으로 6개월 전 사전 예측 (2016년 3분기 경고, 24개월 실적 기준 2017년 1분기. 단일 사례) |
| 수정안 사후 평가 | 권고 10개 시점 평균 상각률 예측 오차(상대) 31%에서 6%로. 2018년 1분기 보유 대출 461,365건 33.5%에서 0.2%로 |
| 운영 방식 비교 (3년, 갱신 승인 가정) | 고정 모형 28.5%, 정기 재학습 18.8% (갱신 13회), 단순 임계값 18.2% (6회), 본 시스템 9.5% (2회, 예측치 조정). 대출 수 가중 평균 27.0%, 22.5%, 21.2%, 11.9% |
| 변별력과 log loss (운영 방식 비교) | 본 시스템 AUC 0.703, log loss 0.366. 정기 재학습 0.701, 0.368 |
| 오류 주입 시험 | 6건 통과 |

- **오차** |실제 상각률 ÷ 예측 상각률 - 1|
- 운영 방식 비교는 분기마다 당시 보유 대출 전체 평가. 같은 대출이 여러 분기에 반복 포함. 판단 이후 신규 대출의 성과 검증 아님
- 사후 평가(기존 모형에 각 시점 권고안 적용)와 운영 방식 비교(갱신을 누적 반영한 3년 운영)는 별도 실험
- 운영 모형 v2 설정과 규칙 r5는 같은 재현 기간 결과를 보고 정한 탐색 단계 결과이며 독립 검증 아님. 상세는 [docs/10_성능개선_r5.md](docs/10_성능개선_r5.md)
- 이전 결과(운영 모형 v1, 규칙 r4)는 `runs/r4`, `results/*_r4.*`에 보존. r4 운영 방식 비교는 조기 기대 함수 재적합 오류가 있던 계산 (docs/10 §4)
- **평가 의견 대응** [docs/09_평가의견_대응.md](docs/09_평가의견_대응.md)

## 4. 폴더
| 경로 | 내용 |
|---|---|
| `index.html`, `app/index.html` | 결과 화면 (`app/simple.html` 원본) |
| `app/detail.html` | 상세 분석 화면 (`app/template.html` 원본) |
| `analysis/` | 자료 정리, 모형, 점검, 판정, 수정안 검증, 재현, 사후 평가, 화면 생성 코드 |
| `runs/<규칙>/<분기>/` | 분기별 판단 근거(`decision.json`), 수정안 비교표 |
| `results/` | 결과 표, 그림, 요약 |
| `docs/` | 단계별 검증 문서 ([목차](docs/README.md)) |
| `ppt/` | 발표 자료 생성 코드(`build_pt.py`, 그리기 도구 `pt_base.py`, 글꼴 `fonts.py`, 그림 `make_assets.py`, PDF 변환 `export.ps1`)와 결과물(`역량PT_김연주.pdf`, `.pptx`, 쪽별 미리보기 `preview_*.png`) |
| `research/cloud/` | 보조 실험 (운영 모형 변형 7종, 수준 보정 9방식 비교). 운영 코드와 분리, 결과 미채택 ([README](research/cloud/README.md)) |
| `tests/` | 누수 방지, 실적 확정 시점, 점검 함수, 수정안 엔진, 보고서 대조 테스트 |

## 5. 주요 코드
| 파일 | 역할 |
|---|---|
| `analysis/m01_build_base.py`, `m02_label_audit.py` | 원천 정리, 상각 정의, 확정 시점, 변수 가용성, 누수 검증 |
| `analysis/m03_baseline_backtest.py` | 기준모형(로지스틱, LightGBM) 학습, 분기별 백테스트 |
| `analysis/pipeline.py` | 분기 처리 함수 (점검, 판정, 진단, 수정안 검증, 근거 기록) |
| `analysis/rules.py`, `candidates.py`, `monitor.py` | 판정 기준, 수정안 검증, 점검 지표 |
| `analysis/scheduler.py`, `m05_replay.py` | 정기 실행(APScheduler), 13개 분기 무인 재현 |
| `analysis/posthoc.py` | 사후 평가 (선택 로직과 분리, 저장된 수정안 사용) |
| `analysis/m11_policies.py`, `m16_policy_weighted.py` | 운영 방식 4종 비교, 평가 대출 수 가중 집계 |
| `analysis/m07_robustness.py`, `m10_injection.py`, `m12_label_timing.py` | 오류 주입 시험, 감지 성능 모의 검증, 확정 시점 가정 민감도 |
| `analysis/m13_model_search.py` | 입력 변수, 튜닝 탐색 (개선 없음, 기존 모형 유지) |
| `analysis/llm_report.py` | LLM 보고서 초안 (조회 도구 5종, 수치, 판정, 표현 자동 검사) |
| `analysis/m14_build_simple.py`, `m08_build_app.py` | 결과 화면, 상세 화면 생성 |
| `analysis/report.py` | 규칙 기반 보고서 문장 생성, 보고서 수치 대조(`verify`) |
| `analysis/config.py`, `data.py`, `labels.py`, `features.py`, `features_v2.py`, `metrics.py`, `models.py`, `inject.py` | 공통 모듈 (경로와 환경변수, 적재, 상각 정의와 확정 시점, 입력 변수, 평가 지표, 모형 학습, 오류 주입) |

## 6. 재실행
```powershell
# 원천 파일: Kaggle Lending Club accepted_2007_to_2018Q4.csv.gz (저장소 미포함)
# 위치: data/raw/ 또는 상위 폴더 data/raw/, 또는 환경변수 CSM_RAW_CSV. 프로젝트 위치 변경 시 CSM_PROJECT_ROOT. Python 3.12
uv pip install -r requirements.txt
python analysis/m01_build_base.py; python analysis/m02_label_audit.py
python analysis/m03_baseline_backtest.py          # 운영 모형 (환경변수 CSM_CHAMPION=v1|v2, 기본 v2)
python analysis/m17_champion_v2.py; python analysis/m18_update_lab.py --scores v2; python analysis/m19_select_lab.py   # 성능 개선 탐색
python analysis/m04_monitor_history.py
python analysis/m05_replay.py                  # 또는 scheduler.py --mode replay
python analysis/posthoc.py --rule r5
python analysis/m11_policies.py; python analysis/m16_policy_weighted.py; python analysis/m07_robustness.py
python analysis/m10_injection.py; python analysis/m12_label_timing.py
python analysis/m08_build_app.py --rule r5; python analysis/m14_build_simple.py --rule r5
python ppt/make_assets.py; python ppt/build_pt.py      # 발표 자료 (기본 규칙 r5)
python -m pytest tests -q
```
- **LLM 보고서(선택)** 환경변수 `ANTHROPIC_API_KEY` 설정 후 `python analysis/llm_report.py --rule r5 --model claude-opus-5-5`
- 난수 SEED 42 고정. 같은 입력, 코드, 규칙으로 재실행 시 기존 결과 재사용

## 7. 규칙 버전
| 버전 | 변경 | 비고 |
|---|---|---|
| r1 | 최초 판정, 수정안 기준 | 2017년 이후 전 분기 갱신 보류 |
| r2 | 수정안 검증 방식 변경 (1년 전 자료로 수립, 이후 실적으로 평가) | 사후 평가 전 고정 |
| r2.1 | 신규 대출 단일 변수 급변 감지, 자료 누락 시 판단 보류 | 오류 주입 시험 반영 |
| r3 | 원인 유형 진단, 문제 집단 탐색 | |
| r4 | 재학습 강화, 추세 반영 보정, 등급별 오차 기준 변경 | r3 사후 평가 확인 후 변경, 이전 결과 보존 |
| r5 (최종) | 운영 모형 v2, 조기 결과 추세 반영 재보정, 수준 차이 10% 이상 시 갱신, 주의 단계 조기 실적 기준 초과 시 실험 | 탐색 단계, docs/10 |

## 8. 한계
- 원천에 상각일 부재, 마지막 납부월 기준 근사 (지연 4, 5, 6개월 가정 시 13개 분기 중 11개 판정 일치)
- 단일 스냅샷 자료로 과거 시점 재현. 실행 시점 변수는 신청 당시 값으로 가정
- 판단 이후 신규 대출의 실적은 자료 범위 밖. 사후 평가는 판단 시점 보유 대출 기준
- 실행 대출 기준 위험. 거절 신청자 포함 전체 신청자로 일반화 불가
- 모형 입력에 Lending Club 등급과 금리 포함. 신규 신청 심사 모형이 아닌 실행 대출의 상각 위험 예측모형
- 규칙 r4, r5와 운영 모형 v2 설정은 같은 기간 결과를 보고 조정. 이후 독립 기간 평가는 자료 범위(2016년 3분기 실행분까지 실적 확정) 한계로 미실시
- 정기 실행(`scheduler.py --mode live`)은 실행마다 적재 자료를 다시 읽음. 원천 수집, 신규 대출 점수 산출, 승인 후 배포, 복구는 미구현
