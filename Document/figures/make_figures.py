# Cal_Beta_Before_train.md의 그림 3개를 생성한다.
# 실행: conda activate sglang && python make_figures.py
import math
import numpy as np
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

INK = "#1F2937"      # 본문 텍스트
MUTED = "#6B7280"    # 보조 텍스트
GRID = "#E5E7EB"
BLUE = "#2563EB"     # 이득/허용 영역
ORANGE = "#EA580C"   # 비용/예산 상한
GRAY = "#9CA3AF"     # floor, 보조선

plt.rcParams.update({
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "figure.facecolor": "white",
    "axes.facecolor": "white", "font.size": 10.5,
})

def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

# toy 분포 (toy_sims/beta_design.py와 동일)
tokens = ["A", "B", "C", "D"]
pi_A = {"A": 0.50, "B": 0.30, "C": 0.15, "D": 0.05}
pi_T = {"A": 0.85, "B": 0.14, "C": 0.008, "D": 0.002}

def project(beta):
    fl = {v: beta * pi_A[v] for v in tokens}
    def bld(c):
        return {v: max(c * pi_T[v], fl[v]) for v in tokens}
    lo, hi = 0.0, 2.0
    for _ in range(200):
        c = (lo + hi) / 2
        if sum(bld(c).values()) > 1.0:
            hi = c
        else:
            lo = c
    return bld(c)

def kl(q, p):
    return sum(q[v] * math.log(q[v] / p[v]) for v in tokens)

# ---------- 그림 1: 이득 곡선 vs 비용 곡선 ----------
betas = np.linspace(0.001, 1.0, 400)
m, N = 0.10, 80
gain = 1 - (1 - m * betas) ** N
cost = np.array([kl(project(b), pi_T) for b in betas])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=160)

ax1.plot(betas, gain * 100, color=BLUE, linewidth=2)
ax1.axvline(0.37, color=GRAY, linestyle="--", linewidth=1.2)
ax1.text(0.385, 25, "무릎 ≈ 0.37", color=MUTED, fontsize=9.5)
for b, lab, dy in [(0.4, "96%", -9), (0.8, "99.9%", -9)]:
    y = (1 - (1 - m * b) ** N) * 100
    ax1.plot([b], [y], "o", color=BLUE, markersize=5)
    ax1.annotate(lab, (b, y), textcoords="offset points", xytext=(4, dy),
                 color=INK, fontsize=9.5)
ax1.set_xlabel("β")
ax1.set_ylabel("발견 확률 (%)")
ax1.set_title("이득 곡선: 무릎에서 꺾여 평평해진다", fontsize=11)
ax1.set_ylim(0, 105)
clean(ax1)

ax2.plot(betas, cost, color=ORANGE, linewidth=2)
ax2.axvline(0.37, color=GRAY, linestyle="--", linewidth=1.2)
ax2.text(0.385, 0.45, "무릎 ≈ 0.37", color=MUTED, fontsize=9.5)
for b in [0.4, 0.8]:
    y = kl(project(b), pi_T)
    ax2.plot([b], [y], "o", color=ORANGE, markersize=5)
    ax2.annotate(f"{y:.2f}", (b, y), textcoords="offset points", xytext=(6, -2),
                 color=INK, fontsize=9.5)
ax2.set_xlabel("β")
ax2.set_ylabel("Cost(β) = KL(q*‖π_T)  (nats)")
ax2.set_title("비용 곡선: 꺾이지 않고 계속 커진다 (toy)", fontsize=11)
clean(ax2)

fig.suptitle("발견 확률: Base 확률 0.10인 branch가 N=80번의 기회에서 한 번이라도 뽑힐 확률",
             fontsize=9.5, color=MUTED, y=0.02, va="bottom")
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("/home/eoeldroal/WorkPlace/ICLR/Document/figures/fig1_gain_vs_cost.png",
            bbox_inches="tight")
plt.close(fig)

# ---------- 그림 2: 두 세계 (min 규칙) ----------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.9), dpi=160)
worlds = [
    ("비용이 싼 세계: 무릎이 먼저 걸린다", 0.90, 0.37, "β = 0.4 (무릎이 정한다)"),
    ("비용이 비싼 세계: 예산이 먼저 걸린다", 0.15, 0.15, "β = 예산이 허락하는 만큼"),
]
for ax, (title, budget, chosen, note) in zip(axes, worlds):
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1)
    allowed = min(0.37, budget)
    ax.axvspan(0, allowed, color=BLUE, alpha=0.10)
    ax.axvline(0.37, color=GRAY, linewidth=1.8)
    ax.text(0.37 + 0.015, 0.86, "무릎 0.37", ha="left", color=MUTED, fontsize=9.5)
    ax.axvline(budget, color=ORANGE, linewidth=1.8)
    bx = budget + 0.015 if budget > 0.5 else budget - 0.015
    bha = "left" if budget > 0.5 else "right"
    ax.text(bx, 0.68, "예산 상한", ha=bha, color=ORANGE, fontsize=9.5)
    ax.plot([chosen - 0.012], [0.42], "o", color=BLUE, markersize=8, clip_on=False)
    ax.annotate(note, (chosen, 0.42), textcoords="offset points", xytext=(8, -4),
                ha="left", color=INK, fontsize=9.5)
    ax.text(allowed / 2, 0.12, "허용 구간", ha="center", color=BLUE, fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("β")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
fig.tight_layout()
fig.savefig("/home/eoeldroal/WorkPlace/ICLR/Document/figures/fig2_two_worlds.png",
            bbox_inches="tight")
plt.close(fig)

# ---------- 그림 3: q* 만들기 (clamp와 재정규화) ----------
toks = ["A", "B", "C (Base가 아끼던 token)"]
teacher = [0.90, 0.099, 0.001]
base = [0.01, 0.05, 0.40]
beta = 0.4
floor = [beta * b for b in base]
c_norm = (1 - 0.16) / (0.90 + 0.099)
qstar = [c_norm * 0.90, c_norm * 0.099, 0.16]

x = np.arange(3)
w = 0.26
fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=160)
b1 = ax.bar(x - w, teacher, w * 0.92, color=BLUE, label="teacher π_T")
b2 = ax.bar(x, floor, w * 0.92, color=GRAY, label="floor = 0.4 × Base")
b3 = ax.bar(x + w, qstar, w * 0.92, color=ORANGE, label="q* (훈련 target)")
for bars in (b1, b2, b3):
    for r in bars:
        ax.annotate(f"{r.get_height():.3f}".rstrip("0").rstrip("."),
                    (r.get_x() + r.get_width() / 2, r.get_height()),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8.5, color=INK)
ax.annotate("teacher(0.001)가 floor(0.16) 아래이므로\nfloor까지 끌어올린다 (clamp)",
            xy=(2 + w - 0.03, 0.10), xytext=(1.42, 0.40), fontsize=9.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1))
ax.text(1.0, 0.72, "A, B는 clamp되지 않음:\n같은 비율(c ≈ 0.84)로 줄여 합을 1로",
        fontsize=9.5, color=MUTED, ha="center")
ax.set_xticks(x, toks)
ax.set_ylabel("확률", rotation=0, labelpad=18)
ax.set_ylim(0, 1.0)
ax.set_title("q* = max(c·π_T, β·π_A) 가 만들어지는 과정 (β=0.4, 가상 예시)", fontsize=11, pad=26)
ax.legend(frameon=False, fontsize=9.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
clean(ax)
fig.tight_layout()
fig.savefig("/home/eoeldroal/WorkPlace/ICLR/Document/figures/fig3_qstar.png",
            bbox_inches="tight")
plt.close(fig)

print("done")
