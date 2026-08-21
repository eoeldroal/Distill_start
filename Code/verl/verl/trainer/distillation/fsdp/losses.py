# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F

from verl.trainer.distillation.projection import relative_floor_target
from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig

# Stands in for log(0) on tokens outside a model's top-k: finite so that max and
# logsumexp stay well-defined, small enough that exp() is exactly zero.
NEG_LOG_PROB = -1e30


def _chunked_topk_log_probs(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute log_softmax(logits).gather(topk_ids) without materializing [B, T, V].

    Uses the identity:
        log_softmax(x).gather(idx) == x.gather(idx) - logsumexp(x, keepdim=True)
    Streams the reduction in chunks of `chunk_size` tokens along (B*T) with fp32
    logsumexp for numerical stability.

    Args:
        logits:    [B, T, V] student logits.
        topk_ids:  [B, T, K] indices to gather.
        chunk_size: number of tokens per chunk; only affects memory, not numerics.

    Returns:
        [B, T, K] tensor with the same dtype as `logits`.
    """
    B, T, V = logits.shape
    K = topk_ids.shape[-1]
    flat_logits = logits.reshape(-1, V)  # [N, V]
    flat_topk = topk_ids.reshape(-1, K)  # [N, K]
    N = flat_logits.shape[0]

    # Edge case: empty input (e.g. fully-padded micro-batch).
    if N == 0:
        return torch.empty((B, T, K), dtype=logits.dtype, device=logits.device)

    out = torch.empty((N, K), dtype=logits.dtype, device=logits.device)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        chunk_logits_fp32 = flat_logits[s:e].float()
        log_z = torch.logsumexp(chunk_logits_fp32, dim=-1, keepdim=True)  # [c, 1]
        chunk_topk_logits = torch.gather(chunk_logits_fp32, dim=-1, index=flat_topk[s:e])
        out[s:e] = (chunk_topk_logits - log_z).to(logits.dtype)
    return out.reshape(B, T, K)


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    # ``use_chunked_topk`` (opt-in, default off) trades latency for memory:
    # the chunked path streams logsumexp + gather to avoid the [B, T, V]
    # log_softmax buffer, enabling long-context (>=64K) where the default
    # F.log_softmax path OOMs. See ``DistillationLossConfig.use_chunked_topk``
    # for trade-offs and benchmark numbers.
    loss_config: DistillationLossConfig = config.distillation_loss
    use_chunked_topk = getattr(loss_config, "use_chunked_topk", False)
    if use_chunked_topk:
        # log_softmax is monotonic, so topk(logits) == topk(log_softmax(logits)).
        student_topk_ids = torch.topk(student_logits, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = _chunked_topk_log_probs(
            student_logits,
            teacher_topk_ids,
            chunk_size=getattr(loss_config, "chunked_topk_chunk_size", 4096),
        )
    else:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }


def compute_relative_floor_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    anchor_topk_log_probs: torch.Tensor,
    anchor_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Compute forward KL against the relative-floor projection target q*.

    Same contract as ``compute_forward_kl_topk``, except the target is q* built
    from teacher and anchor instead of the teacher alone (see
    ``verl.trainer.distillation.projection``). The support is the union of the
    two top-k sets: the anchor's covers what the floor protects, the teacher's
    covers what the teacher wants to keep. Where a model's own top-k does not
    reach, its probability is below that set's smallest entry, so treating it as
    zero is what makes the floor bind there -- which is the intended target,
    since q* = beta*pi_A exactly where the teacher has crushed a candidate.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        anchor_topk_log_probs: (bsz, seqlen, topk).
        anchor_topk_ids: (bsz, seqlen, topk).

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass, teacher_mass, anchor_mass: (bsz, seqlen/sp_size)
    - floor_binding_count, target_floor_mass: (bsz, seqlen/sp_size)
    """
    loss_config: DistillationLossConfig = config.distillation_loss
    beta = loss_config.relative_floor_beta
    assert beta is not None and 0.0 < beta < 1.0, f"relative_floor_beta must be in (0, 1), got {beta}"

    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    assert anchor_topk_log_probs.is_nested and anchor_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)
    anchor_topk_log_probs = anchor_topk_log_probs.values().unsqueeze(0)
    anchor_topk_ids = anchor_topk_ids.values().unsqueeze(0)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
        anchor_topk_log_probs = slice_input_tensor(anchor_topk_log_probs, dim=1)
        anchor_topk_ids = slice_input_tensor(anchor_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]
    assert anchor_topk_log_probs.shape[:2] == anchor_topk_ids.shape[:2] == student_logits.shape[:2]
    topk = teacher_topk_ids.shape[-1]

    # 2. pair the two index sets by binary search, not an all-pairs comparison:
    # the latter costs a [B, T, topk, topk] tensor, which at training scale is
    # larger than the student logits themselves.
    teacher_sorted, teacher_order = teacher_topk_ids.sort(dim=-1)
    slot = torch.searchsorted(teacher_sorted, anchor_topk_ids).clamp(max=topk - 1)
    shared = teacher_sorted.gather(-1, slot) == anchor_topk_ids  # anchor id the teacher also holds

    # 3. lay both models out on the union: teacher's top-k, then the anchor ids the
    # teacher does not hold. Shared ids live in the teacher half only, so their
    # anchor value is scattered there and their anchor-half slot is dropped.
    teacher_lp = teacher_topk_log_probs.float()
    anchor_scratch = teacher_lp.new_full(teacher_lp.shape[:-1] + (topk + 1,), NEG_LOG_PROB)
    anchor_scratch.scatter_(
        -1,
        torch.where(shared, teacher_order.gather(-1, slot), torch.full_like(slot, topk)),
        anchor_topk_log_probs.float(),
    )
    teacher_lp = torch.cat([teacher_lp, teacher_lp.new_full(anchor_topk_log_probs.shape, NEG_LOG_PROB)], dim=-1)
    anchor_lp = torch.cat([anchor_scratch[..., :topk], anchor_topk_log_probs.float()], dim=-1)
    union_ids = torch.cat([teacher_topk_ids, anchor_topk_ids], dim=-1)
    on_support = torch.cat([torch.ones_like(teacher_topk_ids, dtype=torch.bool), ~shared], dim=-1)
    teacher_lp = torch.where(on_support, teacher_lp, NEG_LOG_PROB)
    anchor_lp = torch.where(on_support, anchor_lp, NEG_LOG_PROB)

    # Support-capture diagnostics are read before log_prob_min_clamp, matching the
    # teacher-only path (which measures them at the gather). Reading them after the clamp
    # would add exp(clamp) per off-support-tail slot and push the sums above 1, and the two
    # loss modes would no longer report the same quantity under the same metric name.
    teacher_mass = teacher_lp.exp().sum(dim=-1)
    anchor_mass = anchor_lp.exp().sum(dim=-1)

    # 4. project the teacher onto the floor set to get the target
    if loss_config.log_prob_min_clamp is not None:
        teacher_lp = torch.where(on_support, teacher_lp.clamp_min(loss_config.log_prob_min_clamp), teacher_lp)
        anchor_lp = torch.where(on_support, anchor_lp.clamp_min(loss_config.log_prob_min_clamp), anchor_lp)
    target_log_probs, floor_binding, _ = relative_floor_target(teacher_lp, anchor_lp, beta)
    target_log_probs = torch.where(on_support, target_log_probs, NEG_LOG_PROB)
    floor_binding = floor_binding & on_support

    # Cost(beta) = KL(q* || pi_T): the price the floor pays to pull the target off the
    # teacher. This is the quantity the design's beta rule is written against
    # (beta* = min(beta_knee, Cost^-1(delta))), and it is NOT the training loss below,
    # which is KL(q* || pi_theta) and measures the student instead. Both distributions
    # here are the clamped ones the target was built from, so tokens the teacher truly
    # puts under exp(log_prob_min_clamp) are read as sitting at that value: the metric is
    # a lower bound whose tightness the clamp sets.
    cost_beta = kl_divergence(log_q=teacher_lp, log_p=target_log_probs)

    # 5. student log-probs on the same support. ``use_chunked_topk`` streams the
    # reduction to avoid the [B, T, V] log_softmax buffer, as in the teacher-only path.
    if getattr(loss_config, "use_chunked_topk", False):
        student_topk_log_probs = _chunked_topk_log_probs(
            student_logits, union_ids, chunk_size=getattr(loss_config, "chunked_topk_chunk_size", 4096)
        )
    else:
        student_topk_log_probs = torch.gather(F.log_softmax(student_logits, dim=-1), dim=-1, index=union_ids)
    student_topk_log_probs = student_topk_log_probs.float()
    student_mass = torch.where(on_support, student_topk_log_probs, NEG_LOG_PROB).exp().sum(dim=-1)
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)

    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=target_log_probs)

    # Diagnostics: how much of each distribution the support captured, and whether
    # the floor actually acted (its binding count is the measured k_bind).
    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "anchor_mass": anchor_mass,
        "floor_binding_count": floor_binding.sum(dim=-1).to(distillation_losses.dtype),
        "target_floor_mass": (target_log_probs.exp() * floor_binding).sum(dim=-1),
        "cost_beta": cost_beta,
    }
