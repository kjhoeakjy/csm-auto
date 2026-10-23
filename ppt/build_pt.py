"""역량PT 2장 (표지 제외) → modelops/ppt/역량PT_김연주.pptx
- 1장: 중앙 타임라인 하나에 학업·활동(위)과 프로젝트(아래), BCG X 는 선 위의 가장 큰 이정표
- 2장: 신용평가모형 운영 자동화. 화면 캡처(대표 분기) + 결과 숫자 3개 (results/ 에서 읽음)
- 그리기 도구·글꼴 규칙: ppt/pt_base.py
실행: .venv/Scripts/python modelops/ppt/build_pt.py  → export.ps1 로 PDF·미리보기
"""
import json
import re
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
MO = HERE.parent
sys.path.insert(0, str(MO / "analysis"))
import pt_base as B  # noqa: E402  (그리기 도구)
from pt_base import (F, G1, G2, H, INK, LINE, M, PX, W, WHITE, BAND, circle, header, hline, rect,  # noqa: E402
                   seg, text, vline, ic_bar, ic_building, ic_curve, ic_dots, ic_flag, ic_layers, ic_net)

OUT = HERE / "역량PT_김연주.pptx"
RES = MO / "results"
RULE = sys.argv[1] if len(sys.argv) > 1 else "r4"


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


EXP = [((2021, 3), (2025, 8), ic_cap, "통계학 학사", "통계적 추론과 모델링"),
       ((2024, 9), (2024, 12), ic_chat, "외국인 학생 멘토링", "통계학 전공 학업 지원"),
       ((2025, 1), (2026, 7), ic_people, "인공지능 학회", "학회장, 운영팀장, 스터디 진행, 프로젝트 진행"),
       ((2025, 9), (2027, 2), ic_cap, "통계학 석사", "딥러닝")]
BAR = RGBColor(0xD2, 0xD2, 0xD2)


def tx(ym):
    """연월 → x. 2021.03~2024.09 는 압축, 2024.09~2027.02 는 넓게 (경험이 겹치는 기간 강조)"""
    m = ym[0] * 12 + ym[1]
    a, b, c = 2021 * 12 + 3, 2024 * 12 + 9, 2027 * 12 + 2
    if m <= b:
        return 1.75 + (m - a) / (b - a) * 1.45
    return 3.2 + (m - b) / (c - b) * (XB - 0.1 - 3.2)
PROJ = [("저해상도 조류 이미지 분류", "2025.01 ~ 2025.03", False), ("오늘의집 리뷰 요약 시스템", "2025.03 ~ 2025.08", False),
        ("AutoML 이미지 분류", "2025.05", False), ("담낭암 생존분석 연구 (대형병원)", "2025.06 ~ 2025.08", False),
        ("시각장애인용 피부질환 예측", "2025.07", False), ("환헤지 멀티에이전트", "2025.07 ~ 2025.12", True),
        ("건물별 전력사용량 예측", "2025.08 ~ 2025.09", False), ("3D 의료 파운데이션 모델", "2025.09 ~ 2026.02", True),
        ("의료 데이터 다변량 평균 비교", "2025.10", False), ("확률론 시험 자동 채점", "2026.02 ~ 2026.03", True),
        ("자율주행 ODD 위험평가 에이전트", "2026.05 ~ 2026.06", True)]
Y1, Y2, YM = 3.55, 4.85, 4.2
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
    for i, (st, en, ic, t, k) in enumerate(EXP):  # 학업·경험: 기간 막대(겹치는 기간 표시) + 선 위 아이콘
        x0, x1 = tx(st), tx(en)
        yt = 1.5 + i * 0.47
        text(s, x0, yt, 6.5, 0.22, [[(t, {"weight": "bold", "size": 10.5}),
                                     (f"   {st[0]}.{st[1]:02d} ~ {en[0]}.{en[1]:02d}", {"size": 8.5, "color": G2})]])
        rect(s, x0, yt + 0.27, max(x1 - x0, 0.08), 0.09, BAR)
        seg(s, x0, yt + 0.36, x0, Y1 - 0.23, color=LINE, width=0.75)
        circle(s, x0, Y1, 0.46, fill=WHITE)
        ic(s, x0, Y1)
    bx0, bx1 = tx((2026, 6)), tx((2026, 9))      # BCG 인턴 기간 막대 (석사 막대 아래)
    rect(s, bx0, 3.37, bx1 - bx0, 0.09, BAR)
    text(s, bx1 + 0.08, 3.31, 1.2, 0.22, "BCG 인턴", size=8.5, color=G1)
    x0, x1 = 1.75, 9.0                           # 프로젝트: 시간순 점 + 아래 3단 엇갈림 글
    pos = [(x0 + i * (x1 - x0) / (len(PROJ) - 1), Y2 + 0.32 + (i % 3) * 0.62) for i in range(len(PROJ))]
    for (x, ty) in pos:                          # 연결선을 먼저, 글 상자(흰 바탕)를 위에
        seg(s, x, Y2 + 0.08, x, ty + 0.02, color=G2, width=0.6)
    for (x, ty), (t, p, big) in zip(pos, PROJ):
        circle(s, x, Y2, 0.17 if big else 0.12, fill=INK, line=None)
        fs = 10.5 if big else 8.5
        w = max(len(t) * fs * 0.0118, len(p) * 0.065) + 0.12
        tb = text(s, x - 0.04, ty, w, 0.45, [[(t, {"weight": "bold" if big else "sb", "size": fs})],
                                             [(p, {"size": 8 if big else 7.5, "color": G1 if big else G2})]],
                  spacing=1.2)
        tb.fill.solid(); tb.fill.fore_color.rgb = WHITE
    n0 = len(s.shapes)                           # BCG 인턴 (합류)
    circle(s, XJ, YM, 0.7, fill=INK, line=None)
    ic_brief(s, XJ, YM, c=WHITE)
    bullets = ["데이터마트 제작", "불량 확률 예측", "SHAP 분석 (CTP-CTQ matrix)", "LLM 에이전트 및 인터페이스 개발"]
    lines = [[("BCG 인턴", {"weight": "bold", "size": 12})]] + \
            [[("•  " + b, {"size": 9, "color": G1})] for b in bullets] + \
            [[("2026.06 ~ 2026.09", {"size": 8.5, "color": G2})]]
    text(s, XJ - 0.3, YM - 0.42 - 0.235 * len(lines), 2.9, 0.235 * len(lines), lines, spacing=1.25)
    seg(s, XJ + 0.35, YM, XE - 0.23, YM, color=INK, width=1.5)
    circle(s, XE, YM, 0.46, fill=WHITE, width=1.5)
    ic_card(s, XE, YM)
    text(s, XE - 0.3, YM + 0.36, W - M - XE + 0.3, 0.3, "현대카드", size=13, weight="bold")

# ---------------------------------------------------------------- 2장: 02 New Project
def results():
    """결과 숫자 3개 (results/ 의 계산 결과에서 읽음)"""
    import pandas as pd
    from m08_build_app import kpis as K, load_decisions, load_json
    decs = load_decisions(RULE)
    ph = pd.read_csv(RES / f"m06_posthoc_{RULE}.csv").to_dict("records")
    early_m = int(re.match(r"\d+", K(decs, ph, "2018Q1")[0]["value"]).group())   # 앞당긴 개월 수
    m = next(r for r in ph if r["asof_q"] == "2018Q1" and r["part"] == "전체" and r["candidate"] == "maintain")
    c = next(r for r in ph if r["asof_q"] == "2018Q1" and r["part"] == "전체" and r["is_chosen"])
    s = {r["policy"][0]: r for r in load_json(RES / "m11_policy.json")["summary"]}
    gap = lambda oe: f"{abs(oe - 1) * 100:.0f}%"
    return [(f"{early_m}개월 미리 예측", "상각률 이상 경고 시점",
             "대출 후 1년 중간 실적 활용 (기존 2년 실적)\n2016~2017년 단일 사례"),
            (gap(c['o_e_ratio']), "수정안 적용 후 상각률 예측 오차",
             f"적용 전 {gap(m['o_e_ratio'])}, 상대 오차 (실제÷예측-1)\n"
             f"절대 차이 {abs(m['actual'] - m['pred']) * 100:.1f}%p에서 {abs(c['actual'] - c['pred']) * 100:.1f}%p로\n"
             f"2018년 1분기 보유 대출 {c['n'] / 1e4:.0f}만 건 사후 평가"),
            (f"{s['D']['changes']}회", "3년간 모형 변경 횟수",
             f"재학습 {s['D']['retrains']}, 예측치 조정 {s['D']['changes'] - s['D']['retrains']} (변경 승인 가정)\n"
             f"정기 재학습 {s['B']['changes']}회 대비 평균 오차 {s['B']['book_mean_abs_oe_err'] * 100:.1f}%에서 "
             f"{s['D']['book_mean_abs_oe_err'] * 100:.1f}%로\n"
             f"변별력(AUC)은 정기 재학습 우위 ({s['B']['book_mean_auc']:.3f} 대 {s['D']['book_mean_auc']:.3f})")]


STEPS = ["3개월마다 예측 상각률과 실제 상각률 비교",
         "차이가 기준 초과 시 모형 수정안 11종 성능 비교",
         "기준 통과안 담당자 보고, 적용 여부는 담당자 결정"]
LLM_NOTE = "보고서 초안은 Claude Opus 5.5 작성, 수치는 계산 결과와 항목별 자동 대조"


def slide2(prs, shot=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, M, 0.62, W - 2 * M, 0.45, "02 New Project", size=20, weight="bold")
    text(s, M, 1.12, W - 2 * M, 0.3, "신용평가모형 자동 점검 시스템", size=13, weight="med")
    text(s, M, 1.45, W - 2 * M, 0.25,
         [[("데이터", {"weight": "sb", "color": INK}), ("  Lending Club 대출 226만 건 (2007~2018년)      ", {"color": G1}),
           ("모형", {"weight": "sb", "color": INK}),
           ("  2010~2012년 대출 8.7만 건의 대출 조건, 소득, 신용 이력으로 학습한 LightGBM 모형 (2년 내 상각 확률 예측)", {"color": G1})]],
         size=10)
    hline(s, M, 1.78, W - 2 * M, color=INK)

    lw = 6.0                                    # 왼쪽: 문제와 그림
    text(s, M, 1.92, lw, 0.25, "문제", size=10, color=G1, weight="med")
    text(s, M, 2.12, lw, 0.6, [[("신용평가모형의 예측 상각률과 실제 상각률 차이 확대", {"weight": "bold", "size": 13})],
                                [("실제 상각률 상승, 예측 상각률 하락 (상각률은 대출 실행 후 2년 내 상환 중단 후 손실 처리 비율)", {"size": 10, "color": G1})]],
         spacing=1.35)
    img = HERE / "chart_default.png"
    ih = lw * Image.open(img).height / Image.open(img).width
    s.shapes.add_picture(str(img), Inches(M - 0.05), Inches(2.95), Inches(lw), Inches(ih))

    rx = M + lw + 0.55                          # 오른쪽: 하는 일과 결과
    rw = W - M - rx
    text(s, rx, 1.92, rw, 0.25, "작동 방식", size=10, color=G1, weight="med")
    for i, t in enumerate(STEPS):
        text(s, rx, 2.15 + i * 0.36, rw, 0.3, [[(f"{i + 1}  ", {"weight": "bold", "color": G2}), (t, {})]], size=11)
    text(s, rx + 0.2, 2.15 + 2 * 0.36 + 0.27, rw - 0.2, 0.22, LLM_NOTE, size=8.5, color=G1)
    hline(s, rx, 3.4, rw)
    text(s, rx, 3.5, rw, 0.25, "성과  2015~2018년 자료로 3년간 운영 재현", size=10, color=G1, weight="med")
    for i, (v, lab, note) in enumerate(results()):
        y0 = 3.78 + i * 0.82
        text(s, rx, y0, 2.6, 0.45, v, size=20, weight="bold")
        text(s, rx + 2.65, y0 + 0.02, rw - 2.65, 0.7, [[(lab, {"weight": "sb", "size": 10.5})]] +
             [[(nl, {"size": 9, "color": G1})] for nl in note.split("\n")], spacing=1.25)
    qs = 0.95                                   # QR
    s.shapes.add_picture(str(HERE / "qr.png"), Inches(M), Inches(H - 0.55 - qs), Inches(qs), Inches(qs))
    text(s, M + qs + 0.15, H - 0.55 - qs + 0.18, 4.5, 0.6,
         [[("결과 화면 (시점별 점검 결과 확인)", {"weight": "bold", "size": 11})],
          [("kjhoeakjy.github.io/csm-auto", {"size": 9.5, "color": G1})]], spacing=1.3)
    text(s, rx, H - 0.62, rw, 0.4, [[("판정 규칙은 같은 기간 결과를 보고 조정한 탐색 단계 결과. 상각 시점은 마지막 납부월 기준 근사", {})],
                                  [("시점별 판단은 해당 시점까지 확정된 정보로 수행. 공개 자료 기반 과거 재현, 실제 금융회사 운영 사례 아님", {})]], size=8, color=G2, spacing=1.2)

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
