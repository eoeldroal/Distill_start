"""두 anchor 경로가 같은 target을 만드는지 검증한다.

  (A) anchor_model  : 추론 서버가 sglang topk 로 뽑은 top-k
  (B) anchor_from_ref: 학습 엔진 forward 가 torch.topk 로 뽑은 top-k

둘은 같은 모델·같은 state 이므로 같은 top-k 를 줘야 하고, 따라서 손실도 같아야 한다.
여기서는 (B)의 추출 함수(compute_topk_scores)가 참조 구현과 일치하는지,
그리고 그 출력이 커널에 들어가 (A)와 같은 손실을 내는지를 확인한다.

실행: conda activate ICLR-verl && CUDA_VISIBLE_DEVICES=3 python -m pytest test_anchor_paths_equivalent.py -q -s
"""
import torch
import torch.nn.functional as F

from verl.trainer.distillation.fsdp.losses import compute_relative_floor_topk
from verl.trainer.distillation.losses import compute_topk_scores

DEV = "cuda" if torch.cuda.is_available() else "cpu"
V, K, BETA = 4096, 64, 0.4


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


def test_extraction_matches_reference():
    """compute_topk_scores 가 log_softmax + topk 참조 구현과 일치한다."""
    torch.manual_seed(0)
    logits = torch.randn(1, 32, V, device=DEV) * 4
    out = compute_topk_scores(config=None, distillation_config=_Cfg(), student_logits=logits)
    ref_vals, ref_ids = F.log_softmax(logits.float(), -1).topk(K, -1)
    print(f"\n  logprob 최대 오차 = {float((out['anchor_logprobs'] - ref_vals).abs().max()):.2e}")
    print(f"  id 일치 = {bool((out['anchor_ids'].long() == ref_ids).all())}")
    assert torch.allclose(out["anchor_logprobs"], ref_vals)
    assert bool((out["anchor_ids"].long() == ref_ids).all())


def test_dual_role_call():
    """엔진은 loss_fn 을 두 번 부른다: logits processor 로, 그리고 손실로."""
    logits = torch.randn(1, 8, V, device=DEV)
    as_processor = compute_topk_scores(config=None, distillation_config=_Cfg(), student_logits=logits)
    assert set(as_processor) == {"anchor_logprobs", "anchor_ids"}
    loss, metrics = compute_topk_scores(
        config=None, distillation_config=_Cfg(),
        model_output={"anchor_logprobs": as_processor["anchor_logprobs"]}, data=None,
    )
    print(f"\n  processor 호출 → 키 {sorted(as_processor)}")
    print(f"  손실 호출 → loss={float(loss)}, metrics={metrics}")
    assert float(loss) == 0.0 and metrics == {}


def test_two_paths_give_same_loss():
    """서버가 준 top-k 와 엔진이 뽑은 top-k 가 같으면 손실도 같다."""
    torch.manual_seed(1)
    N = 24
    student_logits = torch.randn(1, N, V, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    anchor_logits = torch.randn(N, V, device=DEV) * 3
    teacher_lp = F.log_softmax(torch.randn(N, V, device=DEV) * 5, -1)
    t_vals, t_ids = teacher_lp.topk(K, -1)

    # (A) 서버 경로: anchor 확률에서 직접 top-k (sglang 이 하는 것과 같은 연산)
    a_vals_server, a_ids_server = F.log_softmax(anchor_logits, -1).topk(K, -1)

    # (B) 엔진 경로: compute_topk_scores 로 추출
    scores = compute_topk_scores(
        config=None, distillation_config=_Cfg(), student_logits=anchor_logits.unsqueeze(0)
    )
    a_vals_engine = scores["anchor_logprobs"][0]
    a_ids_engine = scores["anchor_ids"][0]

    print(f"\n  두 경로의 top-k 값 최대 오차 = {float((a_vals_server - a_vals_engine).abs().max()):.2e}")
    assert bool((a_ids_server == a_ids_engine.long()).all())

    losses = {}
    for name, (av, ai) in (("server", (a_vals_server, a_ids_server)),
                           ("engine", (a_vals_engine, a_ids_engine))):
        out = compute_relative_floor_topk(
            student_logits=student_logits,
            teacher_topk_log_probs=_nested(t_vals), teacher_topk_ids=_nested(t_ids.int()),
            anchor_topk_log_probs=_nested(av), anchor_topk_ids=_nested(ai.int()),
            config=_Cfg(), data_format="bshd")
        losses[name] = float(out["distillation_losses"].mean())
        print(f"  {name} 경로: 손실 {losses[name]:.6f}, "
              f"binding {float(out['floor_binding_count'].mean()):.2f}, "
              f"anchor_mass {float(out['anchor_mass'].mean()):.4f}")
    assert abs(losses["server"] - losses["engine"]) < 1e-6, losses
    print(f"\n  → 두 경로가 동일한 손실을 낸다 (차이 {abs(losses['server']-losses['engine']):.2e})")
