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
RULE = sys.argv[1] if len(sys.argv) > 1 else "r3"


def notes(part):
    p = HERE / "대본.md"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").split(part)[1].split("\n## ")[0].strip()


# ---------------------------------------------------------------- 1장: 두 트랙 → BCG X 에서 합류
STAGES = [("통계 분석", "추론, 생존분석", 0.6, 3.55),
          ("AI 모델 개발", "모델 설계와 실험, 에이전트", 3.55, 7.15),
          ("산업 적용", "데이터 구축과 실사용", 7.15, 9.45)]
STUDY = [(1.45, ic_bar, "통계학 학사", "통계적 추론과 모델링", "2021.03 ~ 2025.08"),
         (3.85, ic_flag, "데이터사이언스 학회", "학회장, 프로젝트 리딩", "2025.01 ~ 2026.06"),
         (6.0, ic_net, "통계학 석사", "딥러닝 모델 설계", "2025.09 ~")]
PROJ = [(2.4, ic_curve, "생존분석 연구보조", "사건 시점 분석", "2025.06 ~ 2026.08"),
        (4.5, ic_dots, "멀티에이전트 시스템 3건", "데이터와 AI 시스템 연결", "2025.07 ~ 2026.06"),
        (6.45, ic_layers, "3D CT 멀티모달 모델", "표현 학습과 평가", "2025.09 ~ 2026.02")]
Y1, Y2, YM = 3.45, 5.45, 4.45                    # 학업·활동 / 프로젝트 / 합류
XJ = 8.3                                         # 합류 = BCG X


def node1(s, x, y, icon, title, kw, period, above):
    circle(s, x, y, 0.46, fill=WHITE)
    icon(s, x, y)
    ty = y - 1.02 if above else y + 0.34
    text(s, x - 0.23, ty, 2.3, 0.75, [[(title, {"weight": "bold", "size": 11})],
                                       [(kw, {"size": 9.5, "color": G1})],
                                       [(period, {"size": 8.5, "color": G2})]], spacing=1.3)


def slide1(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, M, 0.62, W - 2 * M, 0.45, "01 Experiences", size=20, weight="bold")
    hline(s, M, 1.38, W - 2 * M, color=INK)
    yb, hb = 1.55, 0.72
    for i, (name, kw, x0, x1) in enumerate(STAGES):
        rect(s, x0 + (0.02 if i else 0), yb, x1 - x0 - (0.04 if i else 0.02), hb, BAND)
        text(s, x0 + 0.18, yb + 0.12, x1 - x0 - 0.3, 0.5,
             [[(f"{i + 1}  ", {"size": 11, "color": G2, "weight": "med"}), (name, {"size": 12, "weight": "bold"})],
              [(kw, {"size": 9, "color": G1})]], spacing=1.3)
    xb = XJ - 0.75                               # 트랙이 꺾이는 지점
    for y, label, items, above in [(Y1, "학업과 활동", STUDY, True), (Y2, "프로젝트", PROJ, False)]:
        text(s, M, y - 0.27 if above else y + 0.1, 1.0, 0.2, label, size=8.5, color=G2, weight="med")
        hline(s, M, y, xb - M, color=INK, width=1.5)
        seg(s, xb, y, XJ - 0.3, YM + (-0.16 if above else 0.16), color=INK, width=1.5)
        for x, ic, t, k, p in items:
            node1(s, x, y, ic, t, k, p, above)
    n0 = len(s.shapes)
    circle(s, XJ, YM, 0.78, fill=INK, line=None)
    ic_building(s, XJ, YM)
    for shp in list(s.shapes)[n0 + 1:]:          # 검은 원 위 아이콘은 흰색
        shp.line.color.rgb = WHITE
    text(s, XJ - 0.3, YM + 0.5, 2.4, 0.9, [[("BCG X 인턴", {"weight": "bold", "size": 12.5})],
                                            [("데이터마트, 예측 모델, 분석 화면", {"size": 9.5, "color": G1})],
                                            [("2026.06 ~ 2026.09", {"size": 8.5, "color": G2})]], spacing=1.3)
    ex = 10.05                                   # 종점
    seg(s, XJ + 0.39, YM, ex + 0.08, YM, color=INK, width=1.5)
    circle(s, ex + 0.25, YM, 0.34, fill=WHITE, width=1.5)
    text(s, ex + 0.55, YM - 0.27, W - M - ex - 0.55, 0.6,
         [[("현대카드", {"weight": "bold", "size": 14})],
          [("리스크 모형 운영", {"size": 10, "color": G1})]], spacing=1.3)
    s.notes_slide.notes_text_frame.text = notes("## 1장")


# ---------------------------------------------------------------- 2장: 글자 최소
FLOW = ["정기 실행", "점검", "개선안 검증", "보고"]
SHORT = {}


def kpis():
    from m08_build_app import kpis as K, load_decisions
    import pandas as pd
    rule = RULE
    decs = load_decisions(rule)
    ph = pd.read_csv(RES / f"m06_posthoc_{rule}.csv").to_dict("records")
    return K(decs, ph, "2018Q1")


def slide2(prs, shot):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, M, 0.62, W - 2 * M, 0.45, "02 Project", size=20, weight="bold")
    text(s, M, 1.12, W - 2 * M, 0.3, "신용평가모형 운영 자동화", size=13, weight="med")
    hline(s, M, 1.62, W - 2 * M, color=INK)
    cw = (W - 2 * M - 1.6) / 4
    for i, t in enumerate(FLOW):
        x = M + i * cw
        if i:
            B.poly(s, [(x - 0.14, 1.8), (x - 0.06, 1.88), (x - 0.14, 1.96)], color=G2, width=1.0)
        text(s, x + (0.06 if i else 0), 1.76, cw - 0.2, 0.3,
             [[(f"{i + 1}  ", {"size": 11, "color": G2, "weight": "med"}), (t, {"size": 12, "weight": "bold"})]])
    text(s, W - M - 1.7, 1.79, 1.7, 0.3, "모형 변경은 담당자 승인", size=9, color=G1)
    hline(s, M, 2.2, W - 2 * M)

    iw = 7.0
    ih = min(iw * Image.open(shot).height / Image.open(shot).width, 4.55)
    pic = s.shapes.add_picture(str(shot), Inches(M), Inches(2.42), Inches(iw), Inches(ih))
    pic.line.color.rgb = LINE
    pic.line.width = Pt(PX)
    text(s, M, 2.42 + ih + 0.05, iw, 0.2, "2018년 1분기 판단 화면", size=8, color=G2)

    rx = M + iw + 0.5
    rw = W - M - rx
    for i, k in enumerate(kpis()[:3]):
        y0 = 2.42 + i * 1.45
        if i:
            hline(s, rx, y0 - 0.12, rw)
        text(s, rx, y0, rw, 0.22, k["label"], size=9, color=G1, weight="med")
        text(s, rx, y0 + 0.26, rw, 0.55, k["value"], size=30, weight="bold")
        text(s, rx, y0 + 0.9, rw, 0.25, k.get("short", ""), size=9, color=G1)
    text(s, M, H - 0.38, W - 2 * M, 0.2,
         "Lending Club 공개 대출 데이터로 과거 분기를 재현한 프로토타입. 판단에는 당시 데이터만 쓰고, 평가는 이후 확정된 결과로 함",
         size=7.5, color=G2)
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
