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


# ---------------------------------------------------------------- 1장: Experience Timeline
# 선 아이콘 (중심 x, y, 반경 약 0.12 inch)
def ic_cap(s, x, y, c=INK):      # 학사모
    B.poly(s, [(x - 0.13, y - 0.03), (x, y - 0.09), (x + 0.13, y - 0.03), (x, y + 0.03)], closed=True, color=c)
    B.poly(s, [(x - 0.07, y), (x - 0.07, y + 0.07), (x + 0.07, y + 0.07), (x + 0.07, y)], color=c)
    seg(s, x + 0.11, y - 0.02, x + 0.11, y + 0.07, color=c, width=1.0)


def ic_people(s, x, y, c=INK):   # 사람 둘 (학회)
    for dx, sz in [(-0.055, 0.07), (0.06, 0.06)]:
        circle(s, x + dx, y - 0.05, sz, fill=WHITE, line=c, width=1.25)
        B.poly(s, [(x + dx - 0.055, y + 0.09), (x + dx - 0.045, y + 0.03), (x + dx + 0.045, y + 0.03),
                   (x + dx + 0.055, y + 0.09)], color=c, width=1.25)


def ic_chat(s, x, y, c=INK):     # 말풍선 (멘토링)
    B.poly(s, [(x - 0.11, y - 0.08), (x + 0.11, y - 0.08), (x + 0.11, y + 0.04), (x - 0.02, y + 0.04),
               (x - 0.07, y + 0.1), (x - 0.06, y + 0.04), (x - 0.11, y + 0.04)], closed=True, color=c, width=1.25)
    for yy in (-0.035, 0.0):
        seg(s, x - 0.065, y + yy, x + 0.065, y + yy, color=c, width=0.9)


def ic_brief(s, x, y, c=INK):    # 서류가방 (인턴)
    B.poly(s, [(x - 0.13, y - 0.05), (x + 0.13, y - 0.05), (x + 0.13, y + 0.1), (x - 0.13, y + 0.1)],
           closed=True, color=c)
    B.poly(s, [(x - 0.045, y - 0.05), (x - 0.045, y - 0.1), (x + 0.045, y - 0.1), (x + 0.045, y - 0.05)], color=c)
    seg(s, x - 0.13, y + 0.01, x + 0.13, y + 0.01, color=c, width=1.0)


def ic_card(s, x, y, c=INK):     # 카드
    B.poly(s, [(x - 0.12, y - 0.08), (x + 0.12, y - 0.08), (x + 0.12, y + 0.08), (x - 0.12, y + 0.08)],
           closed=True, color=c)
    seg(s, x - 0.12, y - 0.035, x + 0.12, y - 0.035, color=c, width=2.0)
    seg(s, x - 0.08, y + 0.04, x - 0.02, y + 0.04, color=c, width=1.0)


EXP = [(2.0, ic_cap, "통계학 학사", ["통계적 추론과 모델링"], "2021.03 ~ 2025.08"),
       (4.0, ic_chat, "외국인 학생 멘토링", ["통계학 전공 학업 지원"], "2024.09 ~ 2024.12"),
       (6.0, ic_people, "데이터사이언스 학회", ["학회장, 운영팀장", "스터디 진행, 프로젝트 진행"], "2025.01 ~ 2026.07"),
       (8.1, ic_cap, "통계학 석사", ["딥러닝"], "2025.09 ~ 2027.02")]
PROJ = [("민간의료보험과 건강지표 분석", "2023"), ("제주 특산물 가격 예측", "2023"),
        ("스마트팜 참외 수확량 예측", "2024.08 ~ 2024.10"), ("저해상도 조류 이미지 분류", "2025.01 ~ 2025.03"),
        ("오늘의집 리뷰 요약 시스템", "2025.03 ~ 2025.08"), ("AutoML 이미지 분류", "2025.05"),
        ("담낭선암 생존분석 연구", "2025.06 ~ 2025.08"), ("시각장애인용 피부질환 예측", "2025.07"),
        ("환헤지 멀티에이전트", "2025.07 ~ 2025.12"), ("건물별 전력사용량 예측", "2025.08 ~ 2025.09"),
        ("3D 의료 파운데이션 모델", "2025.09 ~ 2026.02"), ("의료 데이터 다변량 평균 비교", "2025.10"),
        ("전차 제원 PCA 분석", "2025.11"), ("확률론 시험 자동 채점", "2026.02 ~ 2026.03"),
        ("자율주행 ODD 위험평가 에이전트", "2026.05 ~ 2026.06")]
Y1, Y2, YM = 2.75, 4.25, 3.5
XB, XJ, XE = 9.35, 9.95, 11.25                   # 꺾임, 합류(BCG), 종점(현대카드)


def slide1(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, M, 0.62, W - 2 * M, 0.45, "Experience Timeline", size=20, weight="bold")
    hline(s, M, 1.3, W - 2 * M, color=INK)
    LX = 1.5                                     # 트랙 시작 (왼쪽 여백에 트랙 이름)
    for y, label, up in [(Y1, "학업, 경험", True), (Y2, "프로젝트", False)]:
        text(s, M, y - 0.1, LX - M - 0.1, 0.2, label, size=9, color=G1, weight="med")
        hline(s, LX, y, XB - LX, color=INK, width=1.5)
        seg(s, XB, y, XJ - 0.28, YM + (-0.14 if up else 0.14), color=INK, width=1.5)
    for x, ic, t, k, p in EXP:                   # 학업·경험: 아이콘 + 위쪽 글
        circle(s, x, Y1, 0.46, fill=WHITE)
        ic(s, x, Y1)
        lines = [[(t, {"weight": "bold", "size": 11})]] + [[(kk, {"size": 9, "color": G1})] for kk in k] + \
                [[(p, {"size": 8.5, "color": G2})]]
        text(s, x - 0.23, Y1 - 0.42 - 0.24 * len(lines), 2.2, 0.24 * len(lines), lines, spacing=1.3)
    x0, x1 = 1.75, 9.0                           # 프로젝트: 시간순 점 + 아래 3단 엇갈림 글
    pos = [(x0 + i * (x1 - x0) / (len(PROJ) - 1), Y2 + 0.32 + (i % 3) * 0.62) for i in range(len(PROJ))]
    for (x, ty) in pos:                          # 연결선을 먼저, 글 상자(흰 바탕)를 위에
        seg(s, x, Y2 + 0.08, x, ty + 0.02, color=G2, width=0.6)
    for (x, ty), (t, p) in zip(pos, PROJ):
        circle(s, x, Y2, 0.12, fill=INK, line=None)
        w = max(len(t) * 0.098, len(p) * 0.06) + 0.12
        tb = text(s, x - 0.04, ty, w, 0.4, [[(t, {"weight": "sb", "size": 8.5})], [(p, {"size": 7.5, "color": G2})]],
                  spacing=1.2)
        tb.fill.solid(); tb.fill.fore_color.rgb = WHITE
    n0 = len(s.shapes)                           # BCG 인턴 (합류)
    circle(s, XJ, YM, 0.7, fill=INK, line=None)
    ic_brief(s, XJ, YM, c=WHITE)
    text(s, XJ - 0.35, YM - 1.05, 2.0, 0.7, [[("BCG 인턴", {"weight": "bold", "size": 12})],
                                             [("실제 산업 현장에서의 모델링", {"size": 9, "color": G1})],
                                             [("2026.06 ~ 2026.08", {"size": 8.5, "color": G2})]], spacing=1.3)
    seg(s, XJ + 0.35, YM, XE - 0.23, YM, color=INK, width=1.5)
    circle(s, XE, YM, 0.46, fill=WHITE, width=1.5)
    ic_card(s, XE, YM)
    text(s, XE - 0.3, YM + 0.36, W - M - XE + 0.3, 0.5, [[("현대카드", {"weight": "bold", "size": 13})],
                                                         [("리스크 모형 운영", {"size": 9, "color": G1})]],
         spacing=1.3)

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
