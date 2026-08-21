# Cal_Beta_Before_train.md의 실측 결과 그림(fig4~fig7)을 생성한다.
# 실행: conda activate sglang && python make_figures2.py
# 데이터 출처: Experiment/PreAnalysis/outputs/
import math
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

names = {f.name for f in fm.fontManager.ttflist}
for cand in ["NanumGothic", "NanumBarunGothic", "NanumSquareRound", "NanumMyeongjo"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

INK = "#1F2937"; MUTED = "#6B7280"; GRID = "#E5E7EB"
BLUE = "#2563EB"; ORANGE = "#EA580C"; GRAY = "#9CA3AF"
plt.rcParams.update({
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "figure.facecolor": "white",
    "axes.facecolor": "white", "font.size": 10.5,
})

from matplotlib.ticker import FuncFormatter, NullFormatter

def declog(ax):
    # log 축 눈금을 10^-1 대신 0.1처럼 십진으로 (폰트에 U+2212가 없어서)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())

def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "Experiment", "PreAnalysis", "outputs")
SWEEP = [0.1, 0.2, 0.4, 0.8]

# toy 비용 곡선 (fig1과 동일한 4-token toy)
tokens = ["A", "B", "C", "D"]
pi_A = {"A": 0.50, "B": 0.30, "C": 0.15, "D": 0.05}
pi_T = {"A": 0.85, "B": 0.14, "C": 0.008, "D": 0.002}
def project(beta):
    fl = {v: beta * pi_A[v] for v in tokens}
    lo, hi = 0.0, 2.0
    for _ in range(200):
        c = (lo + hi) / 2
        if sum(max(c * pi_T[v], fl[v]) for v in tokens) > 1.0: hi = c
        else: lo = c
    return {v: max(c * pi_T[v], fl[v]) for v in tokens}
def kl(q, p):
    return sum(q[v] * math.log(q[v] / p[v]) for v in tokens)

curve_a = pd.read_csv(f"{OUT}/cost_curve.csv", index_col=0).iloc[:, 0]
curve_t = pd.read_csv(f"{OUT}/cost_curve.teacherstates.csv", index_col=0).iloc[:, 0]
betas_toy = np.linspace(0.02, 0.97, 300)
toy = np.array([kl(project(b), pi_T) for b in betas_toy])

# ---------- 그림 4: 실측 비용 곡선 vs toy ----------
fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=160)
ax.plot(betas_toy, toy, color=GRAY, linestyle="--", linewidth=1.6, label="toy 예측")
ax.plot(curve_a.index, curve_a.values, color=ORANGE, linewidth=2, label="실측 (훈련 가중, anchor state)")
ax.axvline(0.37, color=GRAY, linestyle=":", linewidth=1.2)
ax.text(0.375, 0.0035, "무릎 0.37", color=MUTED, fontsize=9.5)
for b in SWEEP:
    y = curve_a.loc[b]
    ax.plot([b], [y], "o", color=ORANGE, markersize=5)
    ax.annotate(f"{y:.3f}", (b, y), textcoords="offset points", xytext=(6, -4),
                color=INK, fontsize=9.5)
ax.set_yscale("log"); ax.set_ylim(0.002, 0.8); declog(ax)
ax.set_xlabel("β"); ax.set_ylabel("Cost(β)  (nats, log 스케일)")
ax.set_title("실측 비용 곡선은 toy와 같은 자릿수에 있다", fontsize=11)
ax.legend(frameon=False, fontsize=9.5, loc="lower right")
clean(ax)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig4_measured_cost.png"), bbox_inches="tight"); plt.close(fig)

# ---------- 그림 5: 위치별 비용 ----------
df = pd.read_parquet(f"{OUT}/cost_states.parquet")
pp = df[df.beta == 0.4].groupby("pos").cost.mean()
pp1 = df[df.beta == 0.1].groupby("pos").cost.mean()
fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=160)
ax.plot(pp1.index, pp1.values, "o-", color=GRAY, linewidth=1.3, markersize=3, label="β = 0.1")
ax.plot(pp.index, pp.values, "o-", color=ORANGE, linewidth=1.8, markersize=3.5, label="β = 0.4")
ax.annotate("위치 0: 11.8 nats\n(<think> 의례)", (0, pp.loc[0]), textcoords="offset points",
            xytext=(14, -12), color=INK, fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1))
ax.set_xscale("symlog", linthresh=1); ax.set_yscale("log"); declog(ax)
ax.set_xlim(-0.15, 600)
ax.set_xlabel("rollout 안의 위치 (token)"); ax.set_ylabel("평균 비용 (nats, log)")
ax.set_title("비용은 위치 0에 집중된다: 형식 token의 값", fontsize=11)
ax.legend(frameon=False, fontsize=9.5)
clean(ax)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig5_cost_by_position.png"), bbox_inches="tight"); plt.close(fig)

# ---------- 그림 6: state 분포의 괄호 ----------
fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=160)
ax.fill_between(curve_a.index, curve_a.values, curve_t.values, color=BLUE, alpha=0.10,
                label="실제 훈련이 지나는 구간")
ax.plot(curve_a.index, curve_a.values, "o-", color=BLUE, linewidth=1.8, markersize=3.5,
        label="anchor state (훈련 시작)")
ax.plot(curve_t.index, curve_t.values, "^-", color=ORANGE, linewidth=1.8, markersize=4,
        label="teacher state (이동의 상한)")
ax.axvline(0.37, color=GRAY, linestyle=":", linewidth=1.2)
ax.text(0.375, 0.012, "무릎 0.37", color=MUTED, fontsize=9.5)
for s, v, dy in [(curve_a, 0.4, -14), (curve_t, 0.4, 6)]:
    ax.annotate(f"{s.loc[v]:.3f}", (v, s.loc[v]), textcoords="offset points",
                xytext=(6, dy), color=INK, fontsize=9.5)
ax.set_yscale("log"); ax.set_ylim(0.008, 2.5); declog(ax)
ax.set_xlabel("β"); ax.set_ylabel("Cost(β)  (nats, log)")
ax.set_title("비용은 student가 어디까지 이동했는지에 따라 구간을 가진다", fontsize=11)
ax.legend(frameon=False, fontsize=9.5, loc="lower right")
clean(ax)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig6_bracket.png"), bbox_inches="tight"); plt.close(fig)

# ---------- 그림 7: 진입 확정 길이와 발견 확률 ----------
w = np.load(f"{OUT}/window_retention.npz")
m, N = 0.10, 80
fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=160)
for b, color in [(0.4, BLUE), (0.8, ORANGE)]:
    prof = w[f"bind_{b}"].mean(0)          # 위치별 binding 비율
    Ls = np.arange(1, 32)
    prun = []
    for L in Ls:
        k = prof[0] + prof[1:1 + L].sum()  # 위치 0의 mode 선택 + 진입 구간
        E = m * b ** k
        prun.append(1 - (1 - E) ** N)
    ax.plot(Ls, np.array(prun) * 100, "-", color=color, linewidth=2, label=f"β = {b}")
ax.axhline(95, color=GRAY, linestyle="--", linewidth=1.2)
ax.text(24.5, 96.5, "설계 목표 95%", color=MUTED, fontsize=9.5)
prof4 = w["bind_0.4"].mean(0)
for L, dy in [(2, 8), (8, 8), (31, 10)]:
    k = prof4[0] + prof4[1:1 + L].sum()
    y = (1 - (1 - m * 0.4 ** k) ** N) * 100
    ax.plot([L], [y], "o", color=BLUE, markersize=5)
    ax.annotate(f"{y:.0f}%", (L, y), textcoords="offset points", xytext=(4, dy),
                color=INK, fontsize=9.5)
ax.set_xlabel("진입이 확정되는 길이 (token)"); ax.set_ylabel("발견 확률 P_run (%)")
ax.set_title("발견 확률은 진입 확정 길이 하나에 달려 있다 (m=0.10, N=80)", fontsize=11)
ax.set_ylim(0, 105)
ax.legend(frameon=False, fontsize=9.5)
clean(ax)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig7_prun_entry.png"), bbox_inches="tight"); plt.close(fig)

print("done")
