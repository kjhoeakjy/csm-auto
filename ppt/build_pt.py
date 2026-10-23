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
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
MO = HERE.parent
sys.path.insert(0, str(MO / "analysis"))
import pt_base as B  # noqa: E402  (그리기 도구)
from pt_base import (F, G1, G2, H, INK, LINE, M, PX, W, WHITE, BAND, circle, header, hline, rect,  # noqa: E402
                   seg, text, vline, ic_bar, ic_building, ic_curve, ic_dots, ic_flag, ic_layers, ic_net)

OUT = HERE / "역량PT_김연주.pptx"
RES = MO / "results"
RULE = sys.argv[1] if len(sys.argv) > 1 else "r5"


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
    text(s, M, 0.62, W - 2 * M, 0.45, "01 Experience Timeline", size=20, weight="bold")
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
    whole = [r for r in ph if r["part"] == "전체"]
    rec = sorted({r["asof_q"] for r in whole if r["is_chosen"]})           # 수정안 권고 시점
    mt = [next(r for r in whole if r["asof_q"] == q and r["candidate"] == "maintain") for q in rec]
    ch = [next(r for r in whole if r["asof_q"] == q and r["is_chosen"]) for q in rec]
    err = lambda rs: sum(abs(r["o_e_ratio"] - 1) for r in rs) / len(rs) * 100
    s = {r["policy"][0]: r for r in load_json(RES / "m11_policy.json")["summary"]}
    # 성과 2 = 2장 그림(운영 재현)과 같은 실험의 마지막 분기: 갱신 없음(A) 대비 본 시스템(D) 보유 대출 예측 오차
    from config import BACKTEST
    from labels import HORIZON
    from pipeline import latest_labeled_q
    gap = lambda o: f"{abs(o - 1) * 100:.0f}%"
    pq = pd.read_csv(RES / "m11_policy_quarter.csv")
    bq = pd.read_csv(BACKTEST)
    bq = bq[bq.model == "lgbm"].set_index("issue_q")
    last = pq.quarter.iloc[-1]
    T = pd.Period(last, freq="Q")
    qs = [str(q) for q in pd.period_range(latest_labeled_q(T, HORIZON) + 1, min(T, pd.Period("2016Q3", freq="Q")), freq="Q")]
    b = bq.loc[qs]
    act = (b.actual * b.n).sum() / b.n.sum()
    oe = {r["policy"][0]: r["book_oe"] for r in pq[pq.quarter == last].to_dict("records")}
    y, qn = T.year, T.quarter
    auc_txt = ("변별력(AUC) 우위" if s["D"]["book_mean_auc"] > s["B"]["book_mean_auc"] + 0.0005
               else "변별력(AUC) 정기 재학습 우위")
    return [(f"{early_m}개월 전 사전 예측", "상각률 이상 경고 시점",
             "대출 1년 차 실적으로 경고\n기존 방식은 2년 차\n실적 확인 후 경고"),
            (gap(oe["D"]), "모형 갱신 후 상각률 예측 오차",
             f"갱신 안 할 시\n예측 오차 {gap(oe['A'])}\n({y}년 {qn}분기 기준)"),
            (f"{s['D']['changes']}회", "3년간 모형 갱신 횟수",
             f"3년 평균 예측 오차 {s['D']['book_mean_abs_oe_err'] * 100:.1f}%\n"
             f"3개월마다 재학습 시\n{s['B']['changes']}회 갱신,\n예측 오차 {s['B']['book_mean_abs_oe_err'] * 100:.1f}%")]


STEPS = ["3개월마다 결과가 확정된 대출의 예측 상각률과 실제 상각률 비교",
         "차이가 기준 초과 시 모형 수정안 12종 성능 비교",
         "기준 통과안 담당자 보고, 담당자가 적용 여부 결정"]
LLM_NOTE = "보고서 초안은 Claude Opus 5.5 작성, 수치는 계산 결과와 항목별 자동 대조"


def wrap(s, n):
    """한 줄 최대 n 글자 폭(한글 1, 영문·숫자 0.55)으로 띄어쓰기 단위 줄바꿈"""
    w = lambda x: sum(1 if ord(c) > 0x2000 else 0.55 for c in x)
    out, cur = [], ""
    for tok in re.findall(r"\([^)]*\)|\S+", s):          # 괄호 안은 한 덩어리 (괄호 중간 줄바꿈 금지)
        cand = (cur + " " + tok).strip()
        if cur and w(cand) > n:
            out.append(cur)
            cur = tok
        else:
            cur = cand
    return out + ([cur] if cur else [])


def slide2(prs, shot=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, M, 0.62, W - 2 * M, 0.45, "02 New Project", size=20, weight="bold")
    text(s, M, 1.12, W - 2 * M, 0.3, "신용평가모형 자동 점검 시스템", size=13, weight="med")
    text(s, M, 1.45, W - 2 * M, 0.25,
         [[("데이터", {"weight": "sb", "color": INK}), ("  Lending Club 대출 226만 건 (2007~2018년)      ", {"color": G1}),
           ("모형", {"weight": "sb", "color": INK}),
           ("  2010~2013년 상반기 대출 14만 건의 대출 조건, 소득, 신용 이력으로 학습한 LightGBM 모형 (2년 내 상각 확률 예측)", {"color": G1})]],
         size=10)
    hline(s, M, 1.78, W - 2 * M, color=INK)

    lw = 6.9                                    # 왼쪽: 문제와 그림 (그래프 비중 확대)
    text(s, M, 1.92, lw, 0.25, "문제", size=10, color=G1, weight="med")
    text(s, M, 2.12, lw, 0.6, [[("신용평가모형의 예측 상각률과 실제 상각률 차이 확대", {"weight": "bold", "size": 13})],
                                [("상각률: 대출 실행 후 2년 내 상환 중단 후 손실 처리 비율", {"size": 10, "color": G1})]],
         spacing=1.35)
    img = HERE / "chart_effect.png"
    ih = lw * Image.open(img).height / Image.open(img).width
    s.shapes.add_picture(str(img), Inches(M - 0.05), Inches(2.62), Inches(lw), Inches(ih))
    text(s, M, 2.62 + ih + 0.02, lw, 0.25, "판단 시점별 보유 대출의 예측 상각률과 사후 확인된 실제 상각률 (갱신 승인 가정)",
         size=9, color=G1, weight="med", align=PP_ALIGN.CENTER)

    rx = M + lw + 0.45                          # 오른쪽: 하는 일과 결과
    rw = W - M - rx
    text(s, rx, 1.92, rw, 0.25, "작동 방식", size=10, color=G1, weight="med")
    for i, t in enumerate(STEPS):
        text(s, rx, 2.15 + i * 0.36, rw, 0.3, [[(f"{i + 1}  ", {"weight": "bold", "color": G2}), (t, {})]], size=11)
    hline(s, rx, 3.4, rw)
    text(s, rx, 3.5, rw, 0.25, "성과  2015~2018년 자료로 3년간 운영 재현", size=10, color=G1, weight="med")
    gap = 0.22                                  # 성과 3개: 가로 카드 (항목명, 수치, 설명)
    cw = (rw - 2 * gap) / 3
    y0 = 3.9
    for i, (v, lab, note) in enumerate(results()):
        x = rx + i * (cw + gap)
        if i:
            vline(s, x - gap / 2, y0, 2.1)
        labs = wrap(lab, 11.5)
        text(s, x, y0, cw, 0.22 * len(labs), [[(ln, {})] for ln in labs], size=9.5, weight="sb", spacing=1.15)
        val = v.replace("전 사전", "전\n사전") if len(v) > 6 else v
        vy = y0 + 0.22 * len(labs) + 0.08
        vh = 0.42 * len(val.split("\n"))
        text(s, x, vy, cw, vh, [[(ln, {})] for ln in val.split("\n")], size=22, weight="bold", spacing=1.0)
        lines = [ln for part in note.split("\n") for ln in wrap(part, 14)]
        text(s, x, vy + vh + 0.12, cw, 1.0, [[(ln, {})] for ln in lines], size=9, color=G1, spacing=1.25)
    qs = 0.8                                    # QR: 제목 줄 오른쪽
    s.shapes.add_picture(str(HERE / "qr.png"), Inches(W - M - qs), Inches(0.42), Inches(qs), Inches(qs))
    text(s, W - M - qs - 3.2, 0.62, 3.05, 0.5,
         [[("시뮬레이션 링크", {"weight": "bold", "size": 10})],
          [("kjhoeakjy.github.io/csm-auto", {"size": 9, "color": G1})]], spacing=1.3, align=PP_ALIGN.RIGHT)
    text(s, rx, H - 0.95, rw, 0.75, [[("판정 규칙과 모형 설정은 같은 기간 결과를 보고 정한 탐색 단계 결과입니다.", {})],
                                   [("상각 시점은 마지막 납부월 기준 근사치입니다.", {})],
                                   [("시점별 판단은 해당 시점까지 확정된 정보로 수행했습니다.", {})],
                                   [("공개 자료 기반 과거 재현이며 실제 금융회사 운영 사례가 아닙니다.", {})]],
         size=8, color=G2, spacing=1.2)

# ---------------------------------------------------------------- 표지 (제출 분량 제외)
def cover(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    rect(s, 0, 0, W, H, INK)                    # 검은 배경
    text(s, M + 0.3, 2.35, 6, 0.9, "김연주", size=40, weight="bold", color=WHITE)
    text(s, M + 0.3, 3.45, 6, 0.6, [[("관심분야", {"size": 11, "color": G2})],
                                     [("리스크 / Collection", {"size": 16, "color": WHITE, "weight": "med"})]], spacing=1.3)
    hline(s, M + 0.3, 4.55, 4.2, color=G1)
    for i, (no, ttl) in enumerate([("01", "Experience Timeline"), ("02", "New Project")]):
        text(s, M + 0.3, 4.8 + i * 0.38, 0.5, 0.3, no, size=14, color=WHITE, weight="bold")
        text(s, M + 0.85, 4.8 + i * 0.38, 5, 0.3, ttl, size=14, color=WHITE)


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
