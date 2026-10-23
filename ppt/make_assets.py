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

qr = qrcode.QRCode(border=1, box_size=20, error_correction=qrcode.constants.ERROR_CORRECT_M)
qr.add_data(URL)
qr.make_image(fill_color="black", back_color="white").save(HERE / "qr.png")
print("ok", URL)
