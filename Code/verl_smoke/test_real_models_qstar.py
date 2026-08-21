"""실제 모델(teacher=Qwen3-4B, anchor=Qwen3-1.7B-Base)로 q*와 Cost(β)를 계산해
새 환경이 사전 분석과 같은 결과를 내는지 확인한다."""
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl.trainer.distillation.fsdp.losses import kl_divergence

TEACHER, ANCHOR, DEV = "Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B-Base", "cuda"
BETAS = [0.1, 0.2, 0.4, 0.8]

def solve_c(pT, pA, beta, iters=60):
    lo = torch.zeros(pT.shape[0], 1, device=pT.device)
    hi = torch.full((pT.shape[0], 1), 5.0, device=pT.device)
    for _ in range(iters):
        c = (lo + hi) / 2
        if_over = torch.maximum(c * pT, beta * pA).sum(-1, keepdim=True) > 1.0
        hi = torch.where(if_over, c, hi); lo = torch.where(if_over, lo, c)
    return (lo + hi) / 2

def main():
    tok = AutoTokenizer.from_pretrained(TEACHER)
    probs = ["If $x+\\frac{1}{x}=5$, compute $x^3+\\frac{1}{x^3}$.",
             "Find the remainder when $2^{100}$ is divided by 7."]
    # Base가 이해하는 plain 형식 (사전 분석에서 확정된 방향)
    texts = [f"Problem: {p}\nSolution: Let's solve this step by step.\n" for p in probs]

    def dists(path):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=DEV).eval()
        out = []
        with torch.no_grad():
            for t in texts:
                ids = tok(t, return_tensors="pt").to(DEV)
                out.append(F.softmax(m(**ids).logits[0, -1].float(), -1))
        del m; torch.cuda.empty_cache()
        return torch.stack(out)

    pT, pA = dists(TEACHER), dists(ANCHOR)
    print(f"vocab={pT.shape[-1]}, state={pT.shape[0]}개\n")
    print(f"{'β':>5} {'Cost(β)':>9} {'clamp된 token':>14} {'floor mass':>11} {'verl kernel 손실':>16}")
    for b in BETAS:
        c = solve_c(pT, pA, b)
        q = torch.maximum(c * pT, b * pA)
        cost = (q * (q / pT.clamp_min(1e-45)).log()).sum(-1)
        clamped = (b * pA >= c * pT)
        fm = (q * clamped).sum(-1)
        # verl의 KL kernel로 student(=anchor를 임시 student로 간주) 대상 손실
        loss = kl_divergence(log_q=pA.clamp_min(1e-45).log(), log_p=q.clamp_min(1e-45).log())
        print(f"{b:>5} {cost.mean():>9.4f} {clamped.sum(-1).float().mean():>14.0f} "
              f"{fm.mean():>11.4f} {loss.mean():>16.4f}")
    print("\n성질 확인")
    c4 = solve_c(pT, pA, 0.4)
    q4 = torch.maximum(c4 * pT, 0.4 * pA)
    print(f"  Σq* = {q4.sum(-1).tolist()}")
    print(f"  c ≥ 1-β 성립: {bool((c4 >= 0.6 - 1e-4).all())}  (c={c4.flatten().tolist()})")
    print(f"  floor 위반 없음: {bool((q4 >= 0.4*pA - 1e-8).all())}")

if __name__ == "__main__":
    main()
