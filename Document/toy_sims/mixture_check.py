import math
tokens = ['A','B','C','D']
pi_A = {'A':0.50,'B':0.30,'C':0.15,'D':0.05}
pi_T = {'A':0.85,'B':0.14,'C':0.008,'D':0.002}
beta = 0.4
floors = {v: beta*pi_A[v] for v in tokens}

# projection (from before)
def project():
    def bld(c): return {v: max(c*pi_T[v], floors[v]) for v in tokens}
    lo,hi = 0.0,2.0
    for _ in range(200):
        c=(lo+hi)/2
        if sum(bld(c).values())>1.0: hi=c
        else: lo=c
    return bld(c)

# arithmetic mixture with alpha = beta (same guaranteed floor)
alpha = beta
q_mix = {v: (1-alpha)*pi_T[v] + alpha*pi_A[v] for v in tokens}
q_star = project()

def kl(q,p): return sum(q[v]*math.log(q[v]/p[v]) for v in tokens)

print(f"{'token':<6}{'teacher':>9}{'floor':>8}{'q* proj':>9}{'q mix':>9}")
for v in tokens:
    print(f"{v:<6}{pi_T[v]:>9.3f}{floors[v]:>8.3f}{q_star[v]:>9.3f}{q_mix[v]:>9.4f}")
print()
print(f"odds A:B  teacher {pi_T['A']/pi_T['B']:.2f} | proj {q_star['A']/q_star['B']:.2f} | mix {q_mix['A']/q_mix['B']:.2f}")
print(f"KL to teacher: proj {kl(q_star,pi_T):.4f} | mix {kl(q_mix,pi_T):.4f}  ({kl(q_mix,pi_T)/kl(q_star,pi_T):.2f}x)")
