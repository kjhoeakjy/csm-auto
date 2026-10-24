"""2장 그림(판단 시점별 보유 대출의 예측 상각률과 사후 확인된 실제 상각률, 흑백) + 결과 화면 주소 QR 코드
실행: .venv/Scripts/python modelops/ppt/make_assets.py
"""
import os
import sys
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

sys.path.insert(0, str(HERE.parent / "analysis"))
from config import BACKTEST                                   # noqa: E402
bt = pd.read_csv(BACKTEST)
# 2장 그림: 판단 시점별 보유 대출의 실제 상각률, 모형 예측(유지), 모형 갱신 시 예측 (운영 방식 비교 m11)
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
fig, ax = plt.subplots(figsize=(7.0, 4.3), dpi=300)
ax.plot(xs, A["act"], color="black", lw=2.8)
ax.plot(xs, A["pred"], color="#8a8a8a", lw=2.4, ls=(0, (4, 3)))
ax.plot(xs, Dp["pred"], color="#444444", lw=2.0, ls=(0, (1, 1.6)), zorder=4)   # 갱신 전 구간은 모형 예측과 같음
chg = [k for k in range(len(Dp)) if Dp.changes_cum[k] > (Dp.changes_cum[k - 1] if k else 0)]
ax.scatter(chg, Dp["pred"][chg], s=34, color="#444444", zorder=5)
for j, k in enumerate(chg):                                  # 모형 갱신 시점 표시 (번호)
    ax.annotate(f"{j + 1}", (k, Dp["pred"][k]), xytext=(9, -17), textcoords="offset points",
                ha="center", fontsize=8, color="#333333")
n = len(A) - 1
ends = sorted([(A["act"][n], "실제", {"fontweight": "bold"}), (Dp["pred"][n], "모형 갱신 시", {"color": "#444444"}),
               (A["pred"][n], "모형 예측", {"color": "#555555"})], key=lambda r: -r[0])
ly = []
for y, _, _ in ends:                                         # 끝 글자 겹침 방지: 위에서부터 최소 0.45%p 간격
    ly.append(min(y, ly[-1] - 0.45) if ly else y)
for (y, lab, kw), yl in zip(ends, ly):
    ax.annotate(f"{lab} {y:.1f}%", (n, y), xytext=(n + 0.25, yl), textcoords="data", ha="left", va="center",
                fontsize=9.5, **kw)
ticks = [k for k, q in enumerate(A.quarter) if q.endswith("Q1")]
ax.set_xticks(ticks, [A.quarter[k][:4] + "년" for k in ticks], fontsize=9, color="#555555")
ax.set_yticks([10, 11, 12, 13, 14], ["10%", "11%", "12%", "13%", "14%"], fontsize=9, color="#555555")
ax.set_ylim(9.7, 14.9)
ax.set_xlim(-0.4, n + 3.4)
ax.hlines([10, 11, 12, 13, 14], -0.4, n + 0.3, color="#e3e3e3", lw=0.8, zorder=0)   # 오른쪽 글자 영역에는 격자선 없음
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