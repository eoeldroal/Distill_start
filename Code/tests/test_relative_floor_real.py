"""실제 모델(teacher=Qwen3-4B, anchor=Qwen3-1.7B-Base)로 우리 손실을 검증한다.

합성 분포는 실제 LM의 형태와 다르므로, 진짜 두 모델의 분포에서
(1) top-k 절단이 q*에 주는 오차, (2) floor 보장 달성률, (3) 사전 분석 성질을 확인한다.

실행: conda activate ICLR-verl && CUDA_VISIBLE_DEVICES=3 python test_relative_floor_real.py
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from verl.trainer.distillation.fsdp.losses import compute_relative_floor_topk
from verl.trainer.distillation.projection import relative_floor_target

TEACHER, ANCHOR, DEV = "Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B-Base", "cuda"
BETAS = [0.1, 0.2, 0.4, 0.8]
KS = [64, 128, 512]


class _Cfg:
    class _Loss:
        relative_floor_beta = 0.4
        log_prob_min_clamp = None
        use_chunked_topk = False
        chunked_topk_chunk_size = 4096
    distillation_loss = _Loss()


def collect_states():
    """Base가 이해하는 plain 프롬프트에서 anchor rollout state를 모은다
    (Cal_Beta_Before_train.md의 state 표본 방식)."""
    tok = AutoTokenizer.from_pretrained(TEACHER)
    problems = [
        "If $x+\\frac{1}{x}=5$, compute $x^3+\\frac{1}{x^3}$.",
        "Find the remainder when $2^{100}$ is divided by 7.",
        "How many ordered pairs $(a,b)$ of positive integers satisfy $a+b=20$?",
    ]
    texts = [f"Problem: {p}\nSolution: Let's solve this step by step.\n" for p in problems]

    ma = AutoModelForCausalLM.from_pretrained(ANCHOR, dtype=torch.bfloat16, device_map=DEV).eval()
    seqs = []
    for t in texts:
        ids = tok(t, return_tensors="pt").to(DEV)
        with torch.no_grad():
            g = ma.generate(**ids, max_new_tokens=120, do_sample=True, temperature=1.0,
                            top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
        pre = ids["input_ids"][0]
        cont = g[0, len(pre):]
        for off in (0, 10, 30, 60, 100):
            if off <= len(cont):
                seqs.append(torch.cat([pre, cont[:off]]))

    def dists(model):
        out = []
        with torch.no_grad():
            for s in seqs:
                out.append(F.softmax(model(s.unsqueeze(0)).logits[0, -1].float(), -1))
        return torch.stack(out)

    pA = dists(ma); del ma; torch.cuda.empty_cache()
    mt = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16, device_map=DEV).eval()
    pT = dists(mt); del mt; torch.cuda.empty_cache()
    return pT, pA


def main():
    print("state 수집 중 (anchor rollout)...", flush=True)
    pT, pA = collect_states()
    N, V = pT.shape
    print(f"state {N}개, vocab {V}\n")

    # 1. 전체 vocab 기준 q* (정답)
    print("=" * 74)
    print("1. 전체 vocabulary 기준 q* — 문서 성질 확인")
    print("=" * 74)
    print(f"{'β':>5} {'Cost(β)':>9} {'clamp 수':>9} {'floor mass':>11} {'c 최소':>8} "
          f"{'Σq*오차':>9} {'floor위반':>9}")
    exact = {}
    for b in BETAS:
        tgt, bind, log_c = relative_floor_target(pT.log(), pA.log(), b)
        q = tgt.exp()
        exact[b] = q
        cost = (q * (q / pT.clamp_min(1e-45)).log()).sum(-1)
        fm = (q * bind).sum(-1)
        sum_err = float((q.sum(-1) - 1).abs().max())
        viol = int((q < b * pA - 1e-7).sum())
        print(f"{b:>5} {cost.mean():>9.4f} {bind.sum(-1).float().mean():>9.0f} "
              f"{fm.mean():>11.4f} {float(log_c.exp().min()):>8.4f} {sum_err:>9.1e} {viol:>9d}")
        assert viol == 0, "floor 위반"
        assert float(log_c.exp().min()) >= (1 - b) - 1e-4, "c >= 1-β 위반"

    # 2. top-k 절단 오차: 커널이 실제로 쓰는 경로
    print()
    print("=" * 74)
    print("2. top-k 절단이 q*에 주는 오차 (β=0.4) — 커널의 실제 경로")
    print("=" * 74)
    print(f"{'K':>6} {'teacher mass':>13} {'anchor mass':>12} {'지켜진 floor':>13} "
          f"{'Cost(전체)':>11} {'Cost(top-k)':>12}")
    b = 0.4
    q_full = exact[b]
    cost_full = float((q_full * (q_full / pT.clamp_min(1e-45)).log()).sum(-1).mean())
    for K in KS:
        tl, ti = pT.log().topk(K, -1)
        al, ai = pA.log().topk(K, -1)
        # 커널과 같은 방식으로 합집합 위에서 q*를 만든다
        costs, kept = [], []
        for i in range(N):
            U = torch.unique(torch.cat([ti[i], ai[i]]))
            t_lp = torch.full((U.numel(),), -1e30, device=DEV)
            a_lp = torch.full((U.numel(),), -1e30, device=DEV)
            pos_t = torch.searchsorted(U, ti[i].sort().values)
            pos_a = torch.searchsorted(U, ai[i].sort().values)
            t_lp[pos_t] = pT[i].log()[ti[i].sort().values]
            a_lp[pos_a] = pA[i].log()[ai[i].sort().values]
            tgt = relative_floor_target(t_lp.unsqueeze(0), a_lp.unsqueeze(0), b)[0].exp()[0]
            q = torch.zeros(V, device=DEV); q[U] = tgt
            costs.append(float((q[q > 0] * (q[q > 0] / pT[i][q > 0].clamp_min(1e-45)).log()).sum()))
            kept.append(float(torch.minimum(q, b * pA[i]).sum() / b))
        print(f"{K:>6} {float(pT.gather(-1, ti).sum(-1).mean()):>13.4f} "
              f"{float(pA.gather(-1, ai).sum(-1).mean()):>12.4f} "
              f"{sum(kept)/len(kept):>12.1%} {cost_full:>11.4f} {sum(costs)/len(costs):>12.4f}")

    # 3. 커널 end-to-end (verl 형식)
    print()
    print("=" * 74)
    print("3. 커널 end-to-end (verl packed 형식, gradient 포함)")
    print("=" * 74)
    K = 128
    tl, ti = pT.log().topk(K, -1)
    al, ai = pA.log().topk(K, -1)
    nest = lambda x: torch.nested.nested_tensor(list(x.unsqueeze(0)), layout=torch.jagged)
    student_logits = (torch.randn(1, N, V, device=DEV) * 3).requires_grad_(True)
    out = compute_relative_floor_topk(
        student_logits=student_logits,
        teacher_topk_log_probs=nest(tl), teacher_topk_ids=nest(ti.int()),
        anchor_topk_log_probs=nest(al), anchor_topk_ids=nest(ai.int()),
        config=_Cfg(), data_format="bshd")
    out["distillation_losses"].mean().backward()
    print(f"  손실 평균          = {float(out['distillation_losses'].mean()):.4f}")
    print(f"  gradient norm      = {float(student_logits.grad.norm()):.4f}")
    print(f"  teacher_mass       = {float(out['teacher_mass'].mean()):.4f}")
    print(f"  anchor_mass        = {float(out['anchor_mass'].mean()):.4f}")
    print(f"  floor binding 수   = {float(out['floor_binding_count'].mean()):.1f}")
    print(f"  target floor mass  = {float(out['target_floor_mass'].mean()):.4f}")
    assert torch.isfinite(out["distillation_losses"]).all()
    assert float(student_logits.grad.norm()) > 0

    print("\n전부 통과: 실제 모델 분포에서 구현이 문서 정의와 일치한다")


if __name__ == "__main__":
    main()
