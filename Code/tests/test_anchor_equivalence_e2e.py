"""실제 모델·실제 state 로 두 anchor 경로의 동등성을 검증한다.

훈련 로그 비교는 rollout 이 확률적이라 문장이 달라져 성립하지 않는다
(verl 의 rollout.do_sample 은 sampling_params 로 전달되지 않는다).
그래서 문장을 고정해 두고 두 경로가 같은 target·같은 손실을 내는지 직접 본다.

  (A) anchor_model : 추론 서버가 하는 것 = log_softmax(logits).topk(K)
  (B) anchor_from_ref : compute_topk_scores 가 학습 엔진 forward 에서 뽑는 것

실행: conda activate ICLR-verl && CUDA_VISIBLE_DEVICES=3 python test_anchor_equivalence_e2e.py
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from verl.trainer.distillation.fsdp.losses import compute_relative_floor_topk
from verl.trainer.distillation.losses import compute_topk_scores

TEACHER, ANCHOR, DEV = "Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B-Base", "cuda"
K, BETA = 64, 0.4


class _Cfg:
    class _Loss:
        relative_floor_beta = BETA
        topk = K
        log_prob_min_clamp = None
        use_chunked_topk = False
        chunked_topk_chunk_size = 4096
    distillation_loss = _Loss()


def _nested(x):
    return torch.nested.nested_tensor(list(x.unsqueeze(0)), layout=torch.jagged)


def main():
    tok = AutoTokenizer.from_pretrained(TEACHER)
    text = ("Problem: If $x+\\frac{1}{x}=5$, compute $x^3+\\frac{1}{x^3}$.\n"
            "Solution: Let's solve this step by step.\nFirst, square both sides.\n")
    ids = tok(text, return_tensors="pt").to(DEV)["input_ids"]
    N = ids.shape[1] - 1
    print(f"고정된 문장: {N} 위치\n")

    def full_logits(path):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=DEV).eval()
        with torch.no_grad():
            lg = m(ids).logits[0, :-1].float()
        del m; torch.cuda.empty_cache()
        return lg

    anchor_logits = full_logits(ANCHOR)
    teacher_logits = full_logits(TEACHER)
    student_logits = anchor_logits.clone().unsqueeze(0).to(torch.bfloat16)  # 훈련 시작 시 student=anchor

    t_lp = F.log_softmax(teacher_logits, -1)
    t_vals, t_ids = t_lp.topk(K, -1)

    # (A) 서버가 돌려주는 것과 같은 연산
    a_vals_server, a_ids_server = F.log_softmax(anchor_logits, -1).topk(K, -1)
    # (B) 학습 엔진 경로
    out = compute_topk_scores(config=None, distillation_config=_Cfg(),
                              student_logits=anchor_logits.unsqueeze(0))
    a_vals_engine, a_ids_engine = out["anchor_logprobs"][0], out["anchor_ids"][0]

    print(f"top-k 값 최대 오차 = {float((a_vals_server - a_vals_engine).abs().max()):.3e}")
    print(f"top-k id 완전 일치 = {bool((a_ids_server == a_ids_engine.long()).all())}\n")

    results = {}
    for name, (av, ai) in (("anchor_model (서버)", (a_vals_server, a_ids_server)),
                           ("anchor_from_ref (엔진)", (a_vals_engine, a_ids_engine))):
        o = compute_relative_floor_topk(
            student_logits=student_logits,
            teacher_topk_log_probs=_nested(t_vals), teacher_topk_ids=_nested(t_ids.int()),
            anchor_topk_log_probs=_nested(av), anchor_topk_ids=_nested(ai.int()),
            config=_Cfg(), data_format="bshd")
        results[name] = {k: float(v.float().mean()) for k, v in o.items()}

    keys = list(next(iter(results.values())))
    print(f"{'지표':<24} {'서버 경로':>14} {'엔진 경로':>14} {'차이':>10}")
    print("-" * 66)
    worst = 0.0
    for k in keys:
        a, b = results["anchor_model (서버)"][k], results["anchor_from_ref (엔진)"][k]
        worst = max(worst, abs(a - b))
        print(f"{k:<24} {a:>14.6f} {b:>14.6f} {abs(a-b):>10.2e}")
    print(f"\n최대 차이 = {worst:.3e}")
    assert worst < 1e-6, f"두 경로가 다르다: {worst}"
    print("→ 실제 모델·고정 문장에서 두 anchor 경로가 동일하다")


if __name__ == "__main__":
    main()
