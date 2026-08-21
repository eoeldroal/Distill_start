"""relative-floor projection 구현 검증.

문서(Document/Draft.md §5.1, Cal_Beta_Before_train.md)에 적힌 정의와 성질을
불변식으로 삼아 검사한다. 실행:
    conda activate ICLR-verl
    CUDA_VISIBLE_DEVICES=3 python -m pytest test_relative_floor.py -v -s
"""
import math

import pytest
import torch
import torch.nn.functional as F

from verl.trainer.distillation.fsdp.losses import compute_relative_floor_topk, kl_divergence
from verl.trainer.distillation.projection import relative_floor_target

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BETAS = [0.1, 0.2, 0.4, 0.8]


# ---------------------------------------------------------------- 참조 구현
def reference_qstar(pT, pA, beta, iters=200):
    """확률 공간에서 직접 이분법으로 푼 q*. 구현과 독립적인 기준."""
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        c = (lo + hi) / 2
        if torch.maximum(c * pT, beta * pA).sum(-1).item() > 1.0:
            hi = c
        else:
            lo = c
    c = (lo + hi) / 2
    return torch.maximum(c * pT, beta * pA), c


def peaked(n_state, V, scale, seed):
    """실제 LM처럼 뾰족한 분포를 만든다 (무작위 logits은 지나치게 평평하다)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return F.softmax(torch.randn(n_state, V, generator=g).to(DEV) * scale, dim=-1)


# ---------------------------------------------------------------- 1. 정의
class TestDefinition:
    def test_toy_matches_paper(self):
        """Document/toy_sims/floor_vs_kl.py 의 기준 수치를 재현한다."""
        pA = torch.tensor([[0.50, 0.30, 0.15, 0.05]], device=DEV)
        pT = torch.tensor([[0.85, 0.14, 0.008, 0.002]], device=DEV)
        tgt_lp, binding, log_c = relative_floor_target(pT.log(), pA.log(), 0.4)
        q = tgt_lp.exp()
        cost = float((q * (q / pT).log()).sum())
        print(f"\n  Cost(0.4) = {cost:.4f} (문서 기준 0.0995)")
        assert abs(cost - 0.0995) < 1e-3
        # clamp 되는 것은 C, D 두 개 (문서의 floor 표: A 0.200 < 0.85c, B 0.120 < 0.14c 아님)
        print(f"  clamp된 token 수 = {int(binding.sum())} (문서 기준 2)")
        assert int(binding.sum()) == 2
        # 정규화 상수: 문서의 c = 0.9293
        print(f"  c = {float(log_c.exp()):.4f} (문서 기준 0.9293)")
        assert abs(float(log_c.exp()) - 0.9293) < 1e-3

    def test_odds_preserved_off_floor(self):
        """clamp되지 않은 token 사이에서는 teacher의 상대 선호가 보존된다 (§5.1)."""
        pA = torch.tensor([[0.50, 0.30, 0.15, 0.05]], device=DEV)
        pT = torch.tensor([[0.85, 0.14, 0.008, 0.002]], device=DEV)
        q = relative_floor_target(pT.log(), pA.log(), 0.4)[0].exp()
        odds_T = float(pT[0, 0] / pT[0, 1])
        odds_q = float(q[0, 0] / q[0, 1])
        print(f"\n  teacher A:B = {odds_T:.4f} / q* A:B = {odds_q:.4f}")
        assert abs(odds_q - odds_T) < 1e-3

    @pytest.mark.parametrize("beta", BETAS)
    def test_matches_reference_solver(self, beta):
        """log 공간 구현이 확률 공간 참조 구현과 일치한다."""
        pT, pA = peaked(8, 4096, 6.0, 0), peaked(8, 4096, 4.0, 1)
        got = relative_floor_target(pT.log(), pA.log(), beta)[0].exp()
        worst = 0.0
        for i in range(pT.shape[0]):
            ref, _ = reference_qstar(pT[i], pA[i], beta)
            worst = max(worst, float((got[i] - ref).abs().max()))
        print(f"\n  beta={beta}: 참조 구현과 최대 오차 {worst:.2e}")
        assert worst < 1e-6


# ---------------------------------------------------------------- 2. 성질
class TestProperties:
    @pytest.mark.parametrize("beta", BETAS)
    def test_sums_to_one(self, beta):
        pT, pA = peaked(16, 8192, 6.0, 2), peaked(16, 8192, 4.0, 3)
        q = relative_floor_target(pT.log(), pA.log(), beta)[0].exp()
        err = float((q.sum(-1) - 1).abs().max())
        print(f"\n  beta={beta}: |Σq*-1| 최대 {err:.2e}")
        assert err < 1e-5

    @pytest.mark.parametrize("beta", BETAS)
    def test_floor_satisfied(self, beta):
        """q*(v) >= beta*pi_A(v) 가 모든 token에서 성립한다 (제약 조건 자체)."""
        pT, pA = peaked(16, 8192, 6.0, 4), peaked(16, 8192, 4.0, 5)
        q = relative_floor_target(pT.log(), pA.log(), beta)[0].exp()
        viol = (q < beta * pA - 1e-7).sum().item()
        margin = float((q - beta * pA).min())
        print(f"\n  beta={beta}: floor 위반 {viol}건, 최소 여유 {margin:.2e}")
        assert viol == 0

    @pytest.mark.parametrize("beta", BETAS)
    def test_teacher_lower_bound(self, beta):
        """q*(v) >= (1-beta)*pi_T(v) — Cal_Beta_Before_train.md §8의 하한."""
        pT, pA = peaked(16, 8192, 6.0, 6), peaked(16, 8192, 4.0, 7)
        q, _, log_c = relative_floor_target(pT.log(), pA.log(), beta)
        q = q.exp()
        c = float(log_c.exp().min())
        viol = (q < (1 - beta) * pT - 1e-7).sum().item()
        print(f"\n  beta={beta}: c 최소 {c:.4f} (>= {1-beta:.2f} 이어야), teacher 하한 위반 {viol}건")
        assert c >= (1 - beta) - 1e-4
        assert viol == 0

    @pytest.mark.parametrize("beta", BETAS)
    def test_optimality_vs_mixture(self, beta):
        """같은 floor 보장에서 q*가 arithmetic mixture보다 teacher에 더 가깝다 (최적성)."""
        pT, pA = peaked(32, 8192, 6.0, 8), peaked(32, 8192, 4.0, 9)
        q = relative_floor_target(pT.log(), pA.log(), beta)[0].exp()
        mix = (1 - beta) * pT + beta * pA  # mixture는 beta*pi_A를 내장 floor로 가짐
        kl_q = (q * (q / pT).log()).sum(-1)
        kl_m = (mix * (mix / pT).log()).sum(-1)
        print(f"\n  beta={beta}: KL(q*||T) 평균 {kl_q.mean():.4f} <= KL(mix||T) {kl_m.mean():.4f}")
        assert bool((kl_q <= kl_m + 1e-6).all()), "최적성 부등식 위반"

    def test_monotone_in_beta(self):
        """비용은 beta에 단조 증가한다 (문서 §1.1의 구조)."""
        pT, pA = peaked(16, 8192, 6.0, 10), peaked(16, 8192, 4.0, 11)
        costs = []
        for b in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]:
            q = relative_floor_target(pT.log(), pA.log(), b)[0].exp()
            costs.append(float((q * (q / pT).log()).sum(-1).mean()))
        print(f"\n  Cost: {[f'{c:.4f}' for c in costs]}")
        assert all(costs[i] < costs[i + 1] for i in range(len(costs) - 1))

    def test_beta_to_zero_recovers_teacher(self):
        """beta -> 0 이면 q* -> pi_T (floor 없음 = 보통의 distillation).

        비용은 beta에 대략 비례해 사라진다. 절대 문턱을 두면 분포의 뾰족함에
        따라 자리가 달라지므로, beta를 10배 줄일 때 비용도 그만큼 줄어드는지를
        본다."""
        pT, pA = peaked(8, 4096, 6.0, 12), peaked(8, 4096, 4.0, 13)
        costs = {}
        for b in (1e-3, 1e-4, 1e-5):
            q = relative_floor_target(pT.log(), pA.log(), b)[0].exp()
            costs[b] = float((q * (q / pT).log()).sum(-1).mean())
            print(f"\n  beta={b:g}: Cost = {costs[b]:.2e}")
        # 10배 감소마다 비용도 최소 5배 줄어든다 (선형 수렴)
        assert costs[1e-4] < costs[1e-3] / 5
        assert costs[1e-5] < costs[1e-4] / 5
        assert costs[1e-5] < 1e-3

    def test_deterministic_teacher_format_law(self):
        """teacher가 한 token에 확률 1을 몰고 anchor가 그것을 무시하면
        q*가 그 token에 남기는 확률은 정확히 1-beta (Cal_Beta §5의 산수)."""
        V = 1000
        pT = torch.full((1, V), 1e-12, device=DEV)
        pT[0, 0] = 1.0 - 1e-12 * (V - 1)
        pA = torch.full((1, V), 1.0 / (V - 1), device=DEV)
        pA[0, 0] = 1e-9  # anchor는 teacher의 token을 사실상 무시
        pA = pA / pA.sum()
        for beta in BETAS:
            q = relative_floor_target(pT.log(), pA.log(), beta)[0].exp()
            print(f"\n  beta={beta}: q*(teacher token) = {float(q[0,0]):.4f} (이론값 {1-beta:.2f})")
            assert abs(float(q[0, 0]) - (1 - beta)) < 5e-3


# ---------------------------------------------------------------- 3. verl 통합
def _nested(x):
    return torch.nested.nested_tensor(list(x), layout=torch.jagged)


class _Cfg:
    """DistillationConfig 의 최소 대역 (커널이 읽는 필드만)."""
    class _Loss:
        relative_floor_beta = 0.4
        log_prob_min_clamp = None
        use_chunked_topk = False
        chunked_topk_chunk_size = 4096
    distillation_loss = _Loss()


class TestVerlIntegration:
    def _inputs(self, B=2, T=4, V=2048, K=64, seed=20):
        """verl의 실제 형식을 재현한다.

        커널은 nested tensor를 ``values().unsqueeze(0)`` 로 풀어 (1, total_nnz, K)
        로 만들고, student_logits도 rmpad된 (1, total_nnz, V) 를 받는다
        (transformer_impl.py 가 ``logits_rmpad.unsqueeze(0)`` 로 호출한다).
        """
        g = torch.Generator(device="cpu").manual_seed(seed)
        N = B * T  # total_nnz
        student_logits = (torch.randn(1, N, V, generator=g) * 3).to(DEV).requires_grad_(True)
        pT = F.softmax((torch.randn(B, T, V, generator=g) * 6).to(DEV), -1)
        pA = F.softmax((torch.randn(B, T, V, generator=g) * 4).to(DEV), -1)
        tl, ti = pT.log().topk(K, -1)
        al, ai = pA.log().topk(K, -1)
        pT_flat = pT.reshape(1, N, V)
        pA_flat = pA.reshape(1, N, V)
        return (student_logits, pT_flat, pA_flat,
                _nested(tl), _nested(ti.int()), _nested(al), _nested(ai.int()))

    def test_kernel_runs_and_grads(self):
        sl, pT, pA, tl, ti, al, ai = self._inputs()
        out = compute_relative_floor_topk(
            student_logits=sl, teacher_topk_log_probs=tl, teacher_topk_ids=ti,
            anchor_topk_log_probs=al, anchor_topk_ids=ai, config=_Cfg(), data_format="bshd")
        loss = out["distillation_losses"].mean()
        loss.backward()
        gn = float(sl.grad.norm())
        print(f"\n  손실 {float(loss):.4f}, gradient norm {gn:.4f}")
        print(f"  반환 키: {sorted(out)}")
        for k, v in out.items():
            assert v.shape == sl.shape[:2], f"{k} shape {v.shape} != {sl.shape[:2]}"
            assert torch.isfinite(v).all(), f"{k}에 비유한 값"
        assert gn > 0

    def test_kernel_matches_full_vocab_when_k_is_v(self):
        """K = V 이면 top-k 절단이 없으므로 전체 vocab 계산과 같아야 한다."""
        V = 512
        sl, pT, pA, tl, ti, al, ai = self._inputs(B=1, T=2, V=V, K=V, seed=21)
        out = compute_relative_floor_topk(
            student_logits=sl, teacher_topk_log_probs=tl, teacher_topk_ids=ti,
            anchor_topk_log_probs=al, anchor_topk_ids=ai, config=_Cfg(), data_format="bshd")
        got = out["distillation_losses"]
        # 직접 전체 vocab으로 계산
        tgt = relative_floor_target(pT.log(), pA.log(), 0.4)[0]
        ref = kl_divergence(log_q=F.log_softmax(sl.float(), -1), log_p=tgt)
        err = float((got - ref).abs().max())
        print(f"\n  K=V 에서 전체 vocab 계산과 최대 오차 {err:.2e}")
        assert err < 2e-3

    def test_floor_binding_diagnostics(self):
        """floor가 실제로 걸린 곳을 세는 진단이 의미 있는 값을 준다."""
        sl, pT, pA, tl, ti, al, ai = self._inputs(seed=22)
        out = compute_relative_floor_topk(
            student_logits=sl, teacher_topk_log_probs=tl, teacher_topk_ids=ti,
            anchor_topk_log_probs=al, anchor_topk_ids=ai, config=_Cfg(), data_format="bshd")
        print(f"\n  floor binding 수 평균 {out['floor_binding_count'].mean():.1f}")
        print(f"  floor가 든 target mass 평균 {out['target_floor_mass'].mean():.4f}")
        print(f"  teacher_mass {out['teacher_mass'].mean():.4f}, anchor_mass {out['anchor_mass'].mean():.4f}")
        assert (out["floor_binding_count"] > 0).any(), "floor가 아무 데서도 걸리지 않았다"
        assert bool((out["target_floor_mass"] >= 0).all())

    def test_duplicate_ids_handled(self):
        """teacher와 anchor의 top-k가 겹칠 때 target이 여전히 분포다."""
        V, K = 256, 128   # K가 커서 겹침이 많이 생기는 설정
        sl, pT, pA, tl, ti, al, ai = self._inputs(B=1, T=3, V=V, K=K, seed=23)
        out = compute_relative_floor_topk(
            student_logits=sl, teacher_topk_log_probs=tl, teacher_topk_ids=ti,
            anchor_topk_log_probs=al, anchor_topk_ids=ai, config=_Cfg(), data_format="bshd")
        assert torch.isfinite(out["distillation_losses"]).all()
        print(f"\n  겹침 많은 설정(V={V},K={K})에서도 손실 유한: "
              f"{out['distillation_losses'].mean():.4f}")


# ---------------------------------------------------------------- 4. config 검증
class TestConfigValidation:
    def test_beta_required(self):
        from verl.workers.config.distillation import DistillationLossConfig
        with pytest.raises(ValueError, match="relative_floor_beta must be set"):
            DistillationLossConfig(loss_mode="relative_floor_topk", use_policy_gradient=False)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_beta_range(self, bad):
        from verl.workers.config.distillation import DistillationLossConfig
        with pytest.raises(ValueError, match="must be in"):
            DistillationLossConfig(loss_mode="relative_floor_topk",
                                   relative_floor_beta=bad, use_policy_gradient=False)

    def test_policy_gradient_rejected(self):
        from verl.workers.config.distillation import DistillationLossConfig
        with pytest.raises(ValueError, match="supervised distillation loss"):
            DistillationLossConfig(loss_mode="relative_floor_topk",
                                   relative_floor_beta=0.4, use_policy_gradient=True)

    def test_valid_config(self):
        from verl.workers.config.distillation import DistillationLossConfig
        c = DistillationLossConfig(loss_mode="relative_floor_topk",
                                   relative_floor_beta=0.4, use_policy_gradient=False)
        assert c.loss_settings.use_topk
        print(f"\n  정상 config: beta={c.relative_floor_beta}, use_topk={c.loss_settings.use_topk}")
