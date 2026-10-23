"""2장용 그림: 예측 상각률 대 실제 상각률(흑백), 학습 기간 표시 + 화면 주소 QR 코드
실행: .venv/Scripts/python modelops/ppt/make_assets.py
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import qrcode

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
URL = "https://kjhoeakjy.github.io/csm-auto/"

fd = Path(os.environ["TEMP"]) / "pretendard_static"
for f in fd.glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Pretendard" if any(fd.glob("*.ttf")) else "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

bt = pd.read_csv(RES / "m03_backtest_quarter.csv")
d = bt[(bt.model == "lgbm") & (bt.issue_q >= "2012Q1")].reset_index(drop=True)
x = range(len(d))
fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=300)
k_tr = d.index[d.issue_q == "2012Q4"][0]               # 학습 기간(2010~2012) 끝, 이후 시간외 평가
ax.axvspan(-0.3, k_tr + 0.5, color="#f0f0f0", lw=0, zorder=0)
ax.text(k_tr / 2, 14.6, "학습 기간", ha="center", fontsize=8.5, color="#555555")
ax.text(k_tr + 1.0, 14.6, "이후 평가 기간", ha="left", fontsize=8.5, color="#555555")
ax.plot(x, d["actual"] * 100, color="black", lw=2.6)
ax.plot(x, d["pred"] * 100, color="#8a8a8a", lw=2.2, ls=(0, (4, 3)))
i = d.index[d.issue_q == "2016Q2"][0]
ax.annotate(f"실제 {d.actual[i] * 100:.1f}%", (i, d.actual[i] * 100), xytext=(-8, 8), textcoords="offset points",
            ha="right", fontsize=10.5, fontweight="bold")
ax.annotate(f"모형 예측 {d.pred[i] * 100:.1f}%", (i, d.pred[i] * 100), xytext=(-8, -16), textcoords="offset points",
            ha="right", fontsize=10.5, color="#555555")
ticks = [k for k, q in enumerate(d.issue_q) if q.endswith("Q1")]
ax.set_xticks(ticks, [d.issue_q[k][:4] + "년" for k in ticks], fontsize=9, color="#555555")
ax.set_yticks([8, 10, 12, 14], ["8%", "10%", "12%", "14%"], fontsize=9, color="#555555")
ax.set_ylim(7.5, 15)
ax.grid(axis="y", color="#e3e3e3", lw=0.8)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#999999")
ax.tick_params(length=0)
ax.set_xlabel("대출 실행 시기", fontsize=9, color="#555555")
fig.tight_layout()
fig.savefig(HERE / "chart_default.png", transparent=False, facecolor="white")
plt.close(fig)

# 2장 그림: 판단 시점별 보유 대출의 실제 상각률, 모형 예측(유지), 모형 갱신 시 예측 (운영 방식 비교 m11)
import sys                                                    # noqa: E402
sys.path.insert(0, str(HERE.parent / "analysis"))
from labels import HORIZON                                    # noqa: E402
from pipeline import latest_labeled_q                         # noqa: E402
pq = pd.read_csv(RES / "m11_policy_quarter.csv")
A = pq[pq.policy.str.startswith("A")].reset_index(drop=True)
Dp = pq[pq.policy.str.startswith("D")].reset_index(drop=True)
bq = bt[bt.model == "lgbm"].set_index("issue_q")
act = []
for t in A.quarter:                                           # 보유 대출 = 24개월 결과 미확정 실행분 (2016Q3 까지)
    T = pd.Period(t, freq="Q")
    qs = [str(q) for q in pd.period_range(latest_labeled_q(T, HORIZON) + 1, min(T, pd.Period("2016Q3", freq="Q")), freq="Q")]
    b = bq.loc[qs]
    act.append((b.actual * b.n).sum() / b.n.sum() * 100)
A["act"] = act
A["pred"] = A["act"] / A["book_oe"]
Dp["pred"] = A["act"] / Dp["book_oe"]
xs = range(len(A))
fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=300)
ax.plot(xs, A["act"], color="black", lw=2.8)
ax.plot(xs, A["pred"], color="#8a8a8a", lw=2.4, ls=(0, (4, 3)))
ax.plot(xs, Dp["pred"], color="#444444", lw=2.0, ls=(0, (1, 1.6)), zorder=4)   # 갱신 전 구간은 모형 예측과 같음
chg = [k for k in range(len(Dp)) if Dp.changes_cum[k] > (Dp.changes_cum[k - 1] if k else 0)]
ax.scatter(chg, Dp["pred"][chg], s=34, color="#444444", zorder=5)
for j, k in enumerate(chg):                                  # 모형 갱신 시점 표시 (번호)
    ax.annotate(f"{j + 1}", (k, Dp["pred"][k]), xytext=(0, -13 if j == 0 else 7), textcoords="offset points",
                ha="center", fontsize=8, color="#333333")
n = len(A) - 1
for y, lab, kw in [(A["act"][n], "실제", {"fontweight": "bold"}), (Dp["pred"][n], "모형 갱신 시", {"color": "#444444"}),
                   (A["pred"][n], "모형 예측", {"color": "#555555"})]:
    ax.annotate(f"{lab} {y:.1f}%", (n, y), xytext=(8, 0), textcoords="offset points", ha="left", va="center",
                fontsize=9.5, **kw)
ticks = [k for k, q in enumerate(A.quarter) if q.endswith("Q1")]
ax.set_xticks(ticks, [A.quarter[k][:4] + "년" for k in ticks], fontsize=9, color="#555555")
ax.set_yticks([10, 12, 14, 16], ["10%", "12%", "14%", "16%"], fontsize=9, color="#555555")
ax.set_ylim(9.3, 16.5)
ax.set_xlim(-0.4, n + 3.4)
ax.hlines([10, 12, 14, 16], -0.4, n + 0.3, color="#e3e3e3", lw=0.8, zorder=0)   # 오른쪽 글자 영역에는 격자선 없음
ax.set_ylabel("보유 대출 상각률", fontsize=9, color="#555555")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#999999")
ax.tick_params(length=0)
ax.set_xlabel("판단 시점", fontsize=9, color="#555555")
fig.tight_layout()
fig.savefig(HERE / "chart_effect.png", transparent=False, facecolor="white")
plt.close(fig)

qr = qrcode.QRCode(border=1, box_size=20, error_correction=qrcode.constants.ERROR_CORRECT_M)
qr.add_data(URL)
qr.make_image(fill_color="black", back_color="white").save(HERE / "qr.png")
print("ok", URL)