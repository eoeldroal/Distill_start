import math

# Toy state: anchor & teacher (as before)
tokens = ['A','B','C','D']
pi_A = {'A':0.50,'B':0.30,'C':0.15,'D':0.05}
pi_T = {'A':0.85,'B':0.14,'C':0.008,'D':0.002}
G = 16

def project(beta):
    fl = {v: beta*pi_A[v] for v in tokens}
    def bld(c): return {v: max(c*pi_T[v], fl[v]) for v in tokens}
    lo,hi = 0.0,2.0
    for _ in range(200):
        c=(lo+hi)/2
        if sum(bld(c).values())>1.0: hi=c
        else: lo=c
    return bld(c)

def kl(q,p): return sum(q[v]*math.log(q[v]/p[v]) for v in tokens)

# Design-time table: beta -> cost, protection, derived guarantees
print(f"{'beta':>5} {'KL cost':>8} {'E_C':>7} {'p_round(C)':>11} {'P_run(C,T=5)':>13} {'m_min(95%,T=5)':>15} {'m_min(T=10)':>12}")
for beta in [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]:
    q = project(beta)
    cost = kl(q, pi_T)
    E_C = q['C']
    p_round = 1-(1-E_C)**G
    P_run5 = 1-(1-E_C)**(G*5)
    # derived m_min: smallest anchor mass with >=95% run-level discovery
    # need (1-beta*m)^(G*T) <= 0.05  ->  m >= (1-0.05**(1/(G*T)))/beta
    m5  = (1-0.05**(1/(G*5)))/beta
    m10 = (1-0.05**(1/(G*10)))/beta
    print(f"{beta:>5.2f} {cost:>8.4f} {E_C:>7.4f} {p_round:>11.3f} {P_run5:>13.4f} {m5:>15.3f} {m10:>12.3f}")

print()
print("D branch (anchor mass 0.05) at beta=0.5:")
q = project(0.5)
E_D = q['D']
print(f"  E_D={E_D:.4f}, p_round={1-(1-E_D)**G:.3f}, P_run(T=5)={1-(1-E_D)**(G*5):.4f}")
