import math

tokens = ['A', 'B', 'C', 'D']
pi_A = {'A': 0.50, 'B': 0.30, 'C': 0.15, 'D': 0.05}
beta = 0.4
floors = {v: beta * pi_A[v] for v in tokens}

def solve_token(rhs, lam, pa):
    lo, hi = 1e-15, 1.0
    for _ in range(300):
        q = (lo + hi) / 2
        if math.log(q) - lam * pa / q < rhs: lo = q
        else: hi = q
    return q

def penalty_opt(lam, pi_T):
    lo_mu, hi_mu = -30.0, 30.0
    for _ in range(300):
        mu = (lo_mu + hi_mu) / 2
        q = {v: solve_token(math.log(pi_T[v]) - 1 - mu, lam, pi_A[v]) for v in tokens}
        if sum(q.values()) > 1.0: lo_mu = mu
        else: hi_mu = mu
    return q

def project(pi_T):
    def bld(c): return {v: max(c * pi_T[v], floors[v]) for v in tokens}
    lo, hi = 0.0, 2.0
    for _ in range(300):
        c = (lo + hi) / 2
        if sum(bld(c).values()) > 1.0: hi = c
        else: lo = c
    return bld(c)

def tune_lambda(pi_T, target_tok='C'):
    lo_l, hi_l = 0.0, 200.0
    for _ in range(300):
        lam = (lo_l + hi_l) / 2
        if penalty_opt(lam, pi_T)[target_tok] < floors[target_tok]: lo_l = lam
        else: hi_l = lam
    return lam

def norm(d):
    s = sum(d.values()); return {k: v/s for k, v in d.items()}

# --- Experiment 1: fixed lambda=1.253, teacher crushes C progressively harder ---
print("=== Exp 1: fixed lambda=1.253, effective protection vs crush strength ===")
print(f"{'pi_T(C)':>10} {'q_pen(C)':>10} {'floor(C)=0.060':>15}")
for ptc in [0.008, 0.002, 5e-4, 1e-4, 1e-5]:
    pi_T = norm({'A': 0.85, 'B': 0.14, 'C': ptc, 'D': 0.002})
    q = penalty_opt(1.253, pi_T)
    print(f"{ptc:>10.5f} {q['C']:>10.4f} {'':>15}")

# --- Experiment 2: harder crush, retune lambda to hit floor(C); head cost ---
print("\n=== Exp 2: retuned lambda for same protection, head cost grows ===")
print(f"{'pi_T(C)':>10} {'lambda':>8} {'q_pen(A)':>9} {'q_proj(A)':>10} {'oddsAB pen':>11} {'oddsAB proj':>12}")
for ptc in [0.008, 1e-3, 1e-4, 1e-5]:
    pi_T = norm({'A': 0.85, 'B': 0.14, 'C': ptc, 'D': 0.002})
    lam = tune_lambda(pi_T)
    qp = penalty_opt(lam, pi_T)
    qs = project(pi_T)
    print(f"{ptc:>10.5f} {lam:>8.2f} {qp['A']:>9.3f} {qs['A']:>10.3f} "
          f"{qp['A']/qp['B']:>11.2f} {qs['A']/qs['B']:>12.2f}")

# --- Experiment 3: budget-G hit probabilities for soft-loss reasoning ---
print("\n=== Exp 3: hit prob 1-(1-p)^G, G=8 ===")
for p in [0.008, 0.02, 0.06, 0.13, 0.23]:
    print(f"  entry prob {p:.3f} -> hit prob {1-(1-p)**8:.3f}")
