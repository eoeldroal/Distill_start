import math

# Toy example: 4 candidate tokens at one state
# A = mainline solution path, B = second path, C, D = rare branches
tokens = ['A', 'B', 'C', 'D']
pi_A = {'A': 0.50, 'B': 0.30, 'C': 0.15, 'D': 0.05}   # anchor
pi_T = {'A': 0.85, 'B': 0.14, 'C': 0.008, 'D': 0.002} # teacher (crushes C, D)
beta = 0.4                                             # keep >= 40% of anchor mass

floors = {v: beta * pi_A[v] for v in tokens}

# ---------- 1) Relative-floor projection: q*(v) = max(c*pi_T(v), floor(v)) ----------
def project(c):
    return {v: max(c * pi_T[v], floors[v]) for v in tokens}

lo, hi = 0.0, 2.0
for _ in range(200):
    c = (lo + hi) / 2
    s = sum(project(c).values())
    if s > 1.0: hi = c
    else: lo = c
q_star = project(c)

# ---------- 2) Forward-KL penalty: min KL(q||pi_T) + lam * KL(pi_A||q) ----------
# Stationarity: log q(v) - lam*pi_A(v)/q(v) = log pi_T(v) - 1 - mu
def solve_token(rhs, lam, pa):
    lo, hi = 1e-12, 1.0
    for _ in range(200):
        q = (lo + hi) / 2
        if math.log(q) - lam * pa / q < rhs: lo = q
        else: hi = q
    return q

def penalty_opt(lam):
    lo_mu, hi_mu = -20.0, 20.0
    for _ in range(200):
        mu = (lo_mu + hi_mu) / 2
        q = {v: solve_token(math.log(pi_T[v]) - 1 - mu, lam, pi_A[v]) for v in tokens}
        if sum(q.values()) > 1.0: lo_mu = mu
        else: hi_mu = mu
    return q

# Tune lambda so token C gets the same protection as the floor (q(C) = 0.06)
lo_l, hi_l = 0.0, 50.0
for _ in range(200):
    lam = (lo_l + hi_l) / 2
    if penalty_opt(lam)['C'] < floors['C']: lo_l = lam
    else: hi_l = lam
q_pen = penalty_opt(lam)

def kl(q, p):
    return sum(q[v] * math.log(q[v] / p[v]) for v in tokens)

print(f"floors (beta=0.4): " + ", ".join(f"{v}:{floors[v]:.3f}" for v in tokens))
print(f"projection scale c = {c:.4f}, tuned lambda = {lam:.3f}\n")
print(f"{'token':<6}{'anchor':>8}{'teacher':>9}{'floor':>8}{'q* proj':>9}{'q pen':>9}")
for v in tokens:
    print(f"{v:<6}{pi_A[v]:>8.3f}{pi_T[v]:>9.3f}{floors[v]:>8.3f}{q_star[v]:>9.3f}{q_pen[v]:>9.3f}")
print()
print(f"teacher odds A:B = {pi_T['A']/pi_T['B']:.3f}")
print(f"proj    odds A:B = {q_star['A']/q_star['B']:.3f}")
print(f"penalty odds A:B = {q_pen['A']/q_pen['B']:.3f}")
print()
print(f"KL(q* || teacher)   = {kl(q_star, pi_T):.4f}   <- distance from teacher, projection")
print(f"KL(qpen || teacher)  = {kl(q_pen, pi_T):.4f}   <- distance from teacher, penalty (same C protection)")
