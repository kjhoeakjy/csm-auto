"""역량PT 2장 (표지 제외) → modelops/ppt/역량PT_김연주.pptx
- 1장: 중앙 타임라인 하나에 학업·활동(위)과 프로젝트(아래), BCG X 는 선 위의 가장 큰 이정표
- 2장: 신용평가모형 운영 자동화. 화면 캡처(대표 분기) + 결과 숫자 3개 (results/ 에서 읽음)
- 그리기 도구·글꼴 규칙은 이전 PPT(ppt/src/build.py)를 그대로 가져다 씀 (수정하지 않음)
실행: .venv/Scripts/python modelops/ppt/build_pt.py  → export.ps1 로 PDF·미리보기
"""
import json
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
MO = HERE.parent
sys.path.insert(0, str(MO.parent / "ppt" / "src"))
sys.path.insert(0, str(MO / "analysis"))
import build as B  # noqa: E402  (이전 PPT 도구)
from build import (F, G1, G2, H, INK, LINE, M, PX, W, WHITE, BAND, circle, header, hline, rect,  # noqa: E402
                   seg, text, vline, ic_bar, ic_building, ic_curve, ic_dots, ic_flag, ic_layers, ic_net)

OUT = HERE / "역량PT_김연주.pptx"
RES = MO / "results"


def notes(part):
    p = HERE / "대본.md"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").split(part)[1].split("\n## ")[0].strip()


# ---------------------------------------------------------------- 1장
STAGES = [("통계적 분석·검증", "통계적 추론 · 사건 시점 분석 · 검증", 0.6, 3.75),
          ("AI 모델·시스템 개발", "모델 설계·실험 · 에이전트 시스템 · 조직 운영", 3.75, 7.55),
          ("산업 현장 적용", "데이터 구축부터 실제 활용까지", 7.55, 9.75)]
ABOVE = [(1.6, ic_bar, "통계학 학사", "통계적 추론·모델링", "2021.03 ~ 2025.08"),
         (3.55, ic_flag, "데이터사이언스·AI 학회", "학회장 · 프로젝트 리딩·조직 운영", "2025.01 ~ 2026.06"),
         (5.85, ic_net, "통계학 석사 · 딥러닝", "모델 설계·실험", "2025.09 ~")]
BELOW = [(2.6, ic_curve, "생존분석 연구보조", "사건 시점 분석·통계적 검증", "2025.06 ~ 2026.08"),
         (4.35, ic_dots, "멀티에이전트 시스템 3건", "데이터·AI 시스템 통합", "2025.07 ~ 2026.06"),
         (6.45, ic_layers, "3D CT 멀티모달 모델", "모델 설계·실험", "2025.09 ~ 2026.02")]
BCG = (8.65, "BCG X 인턴", "데이터마트 · 예측 모델 · 분석 화면 · LLM 활용", "실제 데이터와 사용자 기반 개발", "2026.06 ~ 2026.09")


def node1(s, x, y, icon, title, kw, period, above):
    circle(s, x, y, 0.46, fill=WHITE)
    icon(s, x, y)
    ty = y - 1.1 if above else y + 0.36
    text(s, x - 0.23, ty, 2.4, 0.75, [[(title, {"weight": "bold", "size": 11})],
                                       [(kw, {"size": 9.5, "color": G1})],
                                       [(period, {"size": 8.5, "color": G2})]], spacing=1.3)
    seg(s, x, y - 0.23 if above else y + 0.23, x, y - 0.34 if above else y + 0.34, color=G2, width=0.75)


def slide1(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "01 경험과 노력", "통계적 검증에서 AI 시스템 구현, 산업 현장 적용으로")
    hline(s, M, 1.38, W - 2 * M, color=INK)
    yb, hb = 1.6, 0.85
    for i, (name, kw, x0, x1) in enumerate(STAGES):
        rect(s, x0 + (0.02 if i else 0), yb, x1 - x0 - (0.04 if i else 0.02), hb, BAND)
        text(s, x0 + 0.18, yb + 0.16, x1 - x0 - 0.3, 0.6,
             [[(f"{i + 1}  ", {"size": 11, "color": G2, "weight": "med"}), (name, {"size": 12.5, "weight": "bold"})],
              [(kw, {"size": 9.5, "color": G1})]], spacing=1.35)
    y = 4.3                                        # 중앙 타임라인
    hline(s, M, y, STAGES[2][3] - M, color=INK, width=1.5)
    text(s, M, y - 0.3, 0.8, 0.2, "학업·활동", size=8.5, color=G2, weight="med")
    text(s, M, y + 0.12, 0.8, 0.2, "프로젝트", size=8.5, color=G2, weight="med")
    for x, ic, t, k, p in ABOVE:
        node1(s, x, y, ic, t, k, p, True)
    for x, ic, t, k, p in BELOW:
        node1(s, x, y, ic, t, k, p, False)
    x, title, kw1, kw2, period = BCG             # 가장 큰 이정표: 선 위에 크게
    circle(s, x, y, 0.78, fill=INK, line=None)
    ic_building(s, x, y)
    for shp in list(s.shapes)[-6:]:              # 검은 원 위 아이콘을 흰색으로
        try:
            shp.line.color.rgb = WHITE
        except Exception:
            pass
    text(s, x - 0.55, y + 0.55, 2.6, 0.9, [[(title, {"weight": "bold", "size": 12.5})],
                                            [(kw1, {"size": 9.5, "color": G1})],
                                            [(kw2, {"size": 9.5, "color": G1})],
                                            [(period, {"size": 8.5, "color": G2})]], spacing=1.3)
    # 종점
    ex = 10.15
    seg(s, STAGES[2][3], y, ex - 0.1, y, color=INK, width=1.5)
    B.poly(s, [(ex - 0.2, y - 0.08), (ex - 0.08, y), (ex - 0.2, y + 0.08)], color=INK, width=1.5)
    text(s, ex + 0.05, y - 1.0, W - M - ex, 2.1,
         [[("통계적 검증", {"weight": "bold", "size": 12})],
          [("AI 시스템 구현", {"weight": "bold", "size": 12})],
          [("산업 현장 적용", {"weight": "bold", "size": 12})],
          [("", {"size": 6})],
          [("→ 금융 모형 운영 자동화", {"weight": "bold", "size": 13})],
          [("2장 프로젝트", {"size": 9, "color": G2})]], spacing=1.35)
    s.notes_slide.notes_text_frame.text = notes("## 1장")


# ---------------------------------------------------------------- 2장
FLOW = [("정기 실행", "그 분기에 알 수 있던 데이터만"),
        ("점검", "신규 대출 · 중간 신호 · 확정 결과"),
        ("이상 시 개선안 검증", "유지·재보정·재학습 10종 비교"),
        ("보고", "상태 · 권고 · 근거 3개, 수치 자동 대조")]


def kpis():
    from m08_build_app import kpis as K, load_decisions
    import pandas as pd
    decs = load_decisions("r3")
    ph = pd.read_csv(RES / "m06_posthoc_r3.csv").to_dict("records")
    return K(decs, ph, "2018Q1")


def slide2(prs, shot):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "02 역량", "신용평가모형의 점검부터 개선안 검증·보고까지 자동화",
           "공개 대출 데이터(Lending Club 226만 건)로 과거 13개 분기를 재현 · 단독 설계·구현")
    text(s, M, 1.45, W - 2 * M, 0.3,
         "신규 대출의 24개월 상각 위험을 예측하는 모형 1개가 시간이 지나 실제와 어긋날 때, 유지·재보정·재학습 중 무엇을 할지 매 분기 자동으로 판단",
         size=11.5, weight="med")
    hline(s, M, 1.88, W - 2 * M, color=INK)
    cw = (W - 2 * M - 1.7) / 4
    for i, (t, d) in enumerate(FLOW):
        x = M + i * cw
        if i:
            B.poly(s, [(x - 0.12, 2.18), (x - 0.04, 2.26), (x - 0.12, 2.34)], color=G2, width=1.0)
        text(s, x + (0.1 if i else 0), 2.03, cw - 0.2, 0.55,
             [[(f"{i + 1}  ", {"size": 11, "color": G2, "weight": "med"}), (t, {"size": 12, "weight": "bold"})],
              [(d, {"size": 9, "color": G1})]], spacing=1.3)
    text(s, W - M - 1.6, 2.1, 1.6, 0.45, [[("변경은", {"size": 9, "color": G1})], [("담당자 승인", {"size": 9, "weight": "bold"})]],
         spacing=1.25)
    hline(s, M, 2.72, W - 2 * M)

    iw = 6.3
    ih = iw * Image.open(shot).height / Image.open(shot).width
    ih = min(ih, 4.1)
    pic = s.shapes.add_picture(str(shot), Inches(M), Inches(2.9), Inches(iw), Inches(ih))
    pic.line.color.rgb = LINE
    pic.line.width = Pt(PX)
    text(s, M, 2.9 + ih + 0.05, iw, 0.2, "실제 생성 화면 · 2018년 1분기 판단", size=8, color=G2)

    rx, rw = M + iw + 0.45, W - M - (M + iw + 0.45)
    for i, k in enumerate(kpis()[:3]):
        y0 = 2.9 + i * 1.12
        if i:
            hline(s, rx, y0 - 0.08, rw)
        text(s, rx, y0, rw, 0.22, k["label"], size=9, color=G1, weight="med")
        text(s, rx, y0 + 0.22, rw, 0.5, k["value"], size=26, weight="bold")
        text(s, rx, y0 + 0.74, rw, 0.3, k["note"], size=8.5, color=G1)
    hline(s, rx, 6.33, rw, color=INK)
    text(s, rx, 6.4, rw, 0.6,
         [[("수행", {"weight": "sb"}), ("  정답·시점 검증 · 모형 개발 · 판정 규칙 · 후보 검증 엔진 · 무인 실행·보고", {"color": G1})],
          [("한계", {"weight": "sb"}), ("  상각 시점은 마지막 납부월 기준 근사 · 판단 후 신규 대출 성과는 데이터 범위 밖", {"color": G1})]],
         size=9, spacing=1.35)
    text(s, M, H - 0.38, W - 2 * M, 0.2,
         "공개 대출 데이터 기반 과거 분기 재현·프로토타입. 실제 금융회사 운영 아님. 결과 확정 시점 가정(상각 지연 4·5·6개월)은 별도 검증. "
         "정책 비교는 각 분기 당시 데이터로 결정하고 이후 확정 결과로 평가", size=7.5, color=G2)
    s.notes_slide.notes_text_frame.text = notes("## 2장")


def main():
    shot = HERE / "app_card.png"
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    prs.core_properties.title = "리스크 역량PT"
    prs.core_properties.author = "김연주"
    slide1(prs)
    slide2(prs, shot)
    prs.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
