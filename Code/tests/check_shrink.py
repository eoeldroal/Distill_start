"""(1) target을 정규화하지 않으면 student가 어디로 가는가
   (2) 실제 모델에서 c가 얼마나 줄어드는가
   (3) 대안(다른 방식으로 자리 만들기)과 비교"""
import torch, torch.nn.functional as F, math
from verl.trainer.distillation.projection import relative_floor_target
DEV="cuda"; torch.set_printoptions(precision=4, sci_mode=False)

print("="*78)
print("1. 정규화하지 않은 target을 주면 student는 어디로 수렴하는가")
print("="*78)
# 정규화 안 한 target p (합이 1을 넘음)
p_un = torch.tensor([0.70, 0.20, 0.12, 0.10, 0.06, 0.04], device=DEV)  # 합 1.22
print(f"  주어진 target 합 = {float(p_un.sum()):.4f} (1을 넘음)")
logits = torch.zeros(6, device=DEV, requires_grad=True)
opt = torch.optim.Adam([logits], lr=0.1)
for _ in range(4000):
    opt.zero_grad()
    (p_un * (p_un.log() - F.log_softmax(logits, -1))).sum().backward()
    opt.step()
learned = F.softmax(logits.detach(), -1)
print(f"  student가 수렴한 곳      : {learned.tolist()}")
print(f"  target을 정규화한 값     : {(p_un/p_un.sum()).tolist()}")
print(f"  최대 차이 = {float((learned - p_un/p_un.sum()).abs().max()):.2e}")
print("  → student의 출력은 softmax라 항상 합이 1이다. 정규화 안 한 target을 줘도")
print("     결국 정규화된 것을 배운다. '줄이지 않는' 선택지는 존재하지 않는다.")

print()
print("="*78)
print("2. 실제 모델에서 c는 얼마나 줄어드는가 (Qwen3-4B teacher / 1.7B-Base anchor)")
print("="*78)
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
probs = ["If $x+\\frac{1}{x}=5$, compute $x^3+\\frac{1}{x^3}$.",
         "Find the remainder when $2^{100}$ is divided by 7.",
         "How many ordered pairs $(a,b)$ of positive integers satisfy $a+b=20$?"]
texts = [f"Problem: {p}\nSolution: Let's solve this step by step.\n" for p in probs]
ma = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B-Base", dtype=torch.bfloat16, device_map=DEV).eval()
seqs=[]
for t in texts:
    ids = tok(t, return_tensors="pt").to(DEV)
    with torch.no_grad():
        g = ma.generate(**ids, max_new_tokens=120, do_sample=True, temperature=1.0,
                        top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
    pre, cont = ids["input_ids"][0], g[0, ids["input_ids"].shape[1]:]
    for off in (0,10,30,60,100):
        if off <= len(cont): seqs.append(torch.cat([pre, cont[:off]]))
def dists(m):
    out=[]
    with torch.no_grad():
        for s in seqs: out.append(F.softmax(m(s.unsqueeze(0)).logits[0,-1].float(),-1))
    return torch.stack(out)
pA = dists(ma); del ma; torch.cuda.empty_cache()
mt = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map=DEV).eval()
pT = dists(mt); del mt; torch.cuda.empty_cache()

print(f"  state {pT.shape[0]}개")
print(f"{'β':>5} {'c 평균':>8} {'c 중앙':>8} {'c 최소':>8} {'하한 1-β':>9} {'teacher 최대 손실':>16}")
for b in (0.1,0.2,0.4,0.8):
    _,_,log_c = relative_floor_target(pT.log(), pA.log(), b)
    c = log_c.exp().flatten()
    print(f"{b:>5} {float(c.mean()):>8.4f} {float(c.median()):>8.4f} {float(c.min()):>8.4f} "
          f"{1-b:>9.2f} {(1-float(c.min()))*100:>15.1f}%")

print()
print("="*78)
print("3. 자리를 만드는 다른 방법들과 비교 (β=0.4, teacher에서 멀어진 거리)")
print("="*78)
b=0.4
q_star,_,_ = relative_floor_target(pT.log(), pA.log(), b)
q_star = q_star.exp()
kl = lambda q: float((q*(q/pT.clamp_min(1e-45)).log()).sum(-1).mean())
# (a) 우리 방식: 공통 비율 c로 축소
print(f"  (a) 공통 비율 c로 축소 (우리 방식)      : KL = {kl(q_star):.4f}")
# (b) mixture: 모든 teacher token을 (1-β)배로 일괄 축소
mix = (1-b)*pT + b*pA
print(f"  (b) mixture (1-β)π_T + βπ_A            : KL = {kl(mix):.4f}")
# (c) 최상위 token 하나에서만 자리를 빼앗기
q_c = pT.clone()
floor = b*pA
need = (floor - q_c).clamp_min(0)
q_c = torch.maximum(q_c, floor)
top = pT.argmax(-1, keepdim=True)
q_c.scatter_(-1, top, (q_c.gather(-1,top) - need.sum(-1,keepdim=True)).clamp_min(1e-12))
q_c = q_c / q_c.sum(-1,keepdim=True)
print(f"  (c) 최상위 token에서만 빼기             : KL = {kl(q_c):.4f}")
print(f"\n  → 같은 floor 보장을 지키는 방법 중 (a)가 teacher에 가장 가깝다 (최적성)")
