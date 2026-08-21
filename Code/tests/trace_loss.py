"""손실 계산을 한 위치에서 단계별로 추적한다 (설명용)."""
import math, torch, torch.nn.functional as F
from verl.trainer.distillation.projection import relative_floor_target, solve_log_c

torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)
BETA = 0.4
V = 8   # 설명용 작은 vocabulary
names = ["Let", "We", "First", "Note", "Since", "Observe", "▁the", "▁a"]

# 한 state에서 세 모델의 분포 (설명용 값)
pT = torch.tensor([0.70, 0.20, 0.06, 0.030, 0.008, 0.001, 0.0007, 0.0003])  # teacher
pA = torch.tensor([0.10, 0.08, 0.30, 0.25, 0.150, 0.100, 0.0150, 0.0050])   # anchor(Base)
pS = torch.tensor([0.40, 0.25, 0.15, 0.100, 0.060, 0.030, 0.0100, 0.0040])  # student(현재)
for x in (pT, pA, pS): x /= x.sum()

print("=" * 96)
print("입력: 한 state에서 세 모델이 다음 token에 주는 확률")
print("=" * 96)
print(f"{'token':>10} {'teacher π_T':>12} {'anchor π_A':>11} {'student π_θ':>12}")
for i, n in enumerate(names):
    print(f"{n:>10} {pT[i]:>12.4f} {pA[i]:>11.4f} {pS[i]:>12.4f}")

K = 4
tl, ti = pT.log().topk(K)
al, ai = pA.log().topk(K)
print(f"\nteacher top-{K} = {[names[i] for i in ti]}")
print(f"anchor  top-{K} = {[names[i] for i in ai]}")

print("\n" + "=" * 96)
print("1단계: 지지집합 = teacher top-k ∪ anchor top-k")
print("=" * 96)
ts, torder = ti.sort()
slot = torch.searchsorted(ts, ai).clamp(max=K-1)
shared = ts.gather(-1, slot) == ai
print(f"anchor의 각 token이 teacher 집합에도 있나: "
      f"{[(names[ai[j]], bool(shared[j])) for j in range(K)]}")
union = torch.cat([ti, ai])
on_support = torch.cat([torch.ones(K, dtype=torch.bool), ~shared])
print(f"\n지지집합 슬롯 {len(union)}개 (앞 {K}=teacher 절반, 뒤 {K}=anchor 절반):")
for j in range(len(union)):
    half = "teacher" if j < K else "anchor "
    print(f"  슬롯{j}: {names[union[j]]:>8} ({half})  사용={'O' if on_support[j] else 'X (중복)'}")

print("\n" + "=" * 96)
print("2단계: 두 모델 값을 지지집합에 배치 (-inf = 그 모델의 top-k 밖)")
print("=" * 96)
NEG = -1e30
t_lp = torch.cat([tl, torch.full((K,), NEG)])
scratch = torch.full((K+1,), NEG)
scratch.scatter_(-1, torch.where(shared, torder.gather(-1, slot), torch.tensor(K)), al)
a_lp = torch.cat([scratch[:K], al])
t_lp = torch.where(on_support, t_lp, torch.tensor(NEG))
a_lp = torch.where(on_support, a_lp, torch.tensor(NEG))
print(f"{'슬롯':>5} {'token':>9} {'π_T':>10} {'π_A':>10} {'floor=β·π_A':>13}")
for j in range(len(union)):
    if not on_support[j]: continue
    t = "0 (모름)" if t_lp[j] < -1e20 else f"{t_lp[j].exp():.4f}"
    a = "0 (모름)" if a_lp[j] < -1e20 else f"{a_lp[j].exp():.4f}"
    f_ = "0" if a_lp[j] < -1e20 else f"{BETA*a_lp[j].exp():.4f}"
    print(f"{j:>5} {names[union[j]]:>9} {t:>10} {a:>10} {f_:>13}")

print("\n" + "=" * 96)
print(f"3단계: 정규화 상수 c 찾기 (Σ max(c·π_T, β·π_A) = 1)")
print("=" * 96)
log_floor = a_lp + math.log(BETA)
lo, hi = math.log1p(-BETA), float(-torch.logsumexp(t_lp, -1))
print(f"  탐색 구간: c ∈ [1-β, 1/Σπ_T] = [{math.exp(lo):.4f}, {math.exp(hi):.4f}]")
log_c = solve_log_c(t_lp.unsqueeze(0), log_floor.unsqueeze(0), BETA)
c = float(log_c.exp())
print(f"  찾은 c = {c:.4f}")

print("\n" + "=" * 96)
print("4단계: target q* = max(c·π_T, β·π_A)")
print("=" * 96)
tgt, binding, _ = relative_floor_target(t_lp.unsqueeze(0), a_lp.unsqueeze(0), BETA)
tgt, binding = tgt[0], binding[0]
q = tgt.exp()
print(f"{'슬롯':>5} {'token':>9} {'c·π_T':>9} {'β·π_A':>9} {'승자':>9} {'q*':>9} {'π_T 대비':>10}")
for j in range(len(union)):
    if not on_support[j]: continue
    ct = c*t_lp[j].exp() if t_lp[j] > -1e20 else 0.0
    fl = BETA*a_lp[j].exp() if a_lp[j] > -1e20 else 0.0
    win = "floor" if binding[j] else "teacher"
    ratio = f"{q[j]/pT[union[j]]:.2f}배" if pT[union[j]] > 1e-9 else "-"
    print(f"{j:>5} {names[union[j]]:>9} {ct:>9.4f} {fl:>9.4f} {win:>9} {q[j]:>9.4f} {ratio:>10}")
print(f"\n  Σq* = {float(q.sum()):.6f}")

print("\n" + "=" * 96)
print("5단계: 손실 = Σ q*(v)·[log q*(v) − log π_θ(v)]")
print("=" * 96)
s_lp = pS.log()[union]
s_lp = torch.where(on_support, s_lp, torch.tensor(NEG))
terms = q * (tgt - s_lp)
print(f"{'슬롯':>5} {'token':>9} {'q*':>8} {'π_θ':>8} {'log(q*/π_θ)':>13} {'기여도':>10}")
for j in range(len(union)):
    if not on_support[j]: continue
    print(f"{j:>5} {names[union[j]]:>9} {q[j]:>8.4f} {pS[union[j]]:>8.4f} "
          f"{float(tgt[j]-s_lp[j]):>13.4f} {float(terms[j]):>10.4f}")
print(f"\n  손실 = {float(terms.sum()):.4f} nats")

print("\n" + "=" * 96)
print("비교: vanilla distillation(teacher만)이라면")
print("=" * 96)
v_terms = pT[ti] * (pT[ti].log() - pS[ti].log())
print(f"  vanilla 손실 = {float(v_terms.sum()):.4f} nats (teacher top-{K}에 대해서만)")
print(f"  우리 손실   = {float(terms.sum()):.4f} nats")
print(f"\n  vanilla는 'First'(anchor가 0.30을 주던 token)를 target에서 {pT[2]:.4f}로 두지만,")
j = int((union == 2).nonzero()[0])
print(f"  우리는 {q[j]:.4f}로 둔다 ({q[j]/pT[2]:.1f}배). 이것이 floor가 하는 일이다.")
