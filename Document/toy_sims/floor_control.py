import math, random

random.seed(7)
beta = 0.4
N_STATES = 200

# Heterogeneous states: head A, mid B, rare branch C
# anchor keeps C alive at 0.03~0.20; teacher crushes C by 1e-5 ~ 1e-2
states = []
for _ in range(N_STATES):
    aA = random.uniform(0.40, 0.70); aC = random.uniform(0.03, 0.20)
    tA = random.uniform(0.80, 0.95); tC = 10 ** random.uniform(-5, -2)
    states.append(({'A': aA, 'B': 1-aA-aC, 'C': aC},
                   {'A': tA, 'B': 1-tA-tC, 'C': tC}))

def solve_token(rhs, lam, pa):
    lo, hi = 1e-15, 1.0
    for _ in range(80):
        q = (lo+hi)/2
        if math.log(q) - lam*pa/q < rhs: lo = q
        else: hi = q
    return q

def penalty_opt(lam, pA, pT):
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mu = (lo+hi)/2
        q = {v: solve_token(math.log(pT[v])-1-mu, lam, pA[v]) for v in pT}
        if sum(q.values()) > 1.0: lo = mu
        else: hi = mu
    return q

def project(pA, pT):
    fl = {v: beta*pA[v] for v in pA}
    def bld(c): return {v: max(c*pT[v], fl[v]) for v in pT}
    lo, hi = 0.0, 2.0
    for _ in range(80):
        c = (lo+hi)/2
        if sum(bld(c).values()) > 1.0: hi = c
        else: lo = c
    return bld(c)

def kl(q, p): return sum(q[v]*math.log(q[v]/p[v]) for v in q)

# projection: protects every state's C by construction
proj_kl = sum(kl(project(pA, pT), pT) for pA, pT in states) / N_STATES

print(f"projection: rare-branch protection 200/200 states, mean KL to teacher = {proj_kl:.4f}\n")
print(f"{'lambda':>8} {'C protected':>12} {'mean KL':>9} {'x proj cost':>12}")
for lam in [0.3, 0.6, 1.0, 1.6, 2.5, 4.0, 6.3, 10.0, 16.0]:
    ok, tot_kl = 0, 0.0
    for pA, pT in states:
        q = penalty_opt(lam, pA, pT)
        if q['C'] >= beta*pA['C']: ok += 1
        tot_kl += kl(q, pT)
    print(f"{lam:>8.1f} {ok:>7}/200 {tot_kl/N_STATES:>9.4f} {tot_kl/N_STATES/proj_kl:>11.1f}x")

# marginal-value illustration: discovery prob under budget G=8
print("\nP_hit(G=8) = 1-(1-E)^8:")
for e in [0.005, 0.02, 0.06, 0.73]:
    print(f"  E={e:<6} -> {1-(1-e)**8:.4f}")
