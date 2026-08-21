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

"""Relative-floor projection: the distillation target used by ``relative_floor_topk``.

Engine-agnostic math, kept out of the strategy kernels so fsdp and (later)
megatron share one implementation. Everything is in log space because teacher
probabilities on crushed tokens underflow float32, and those tokens are exactly
where the floor is meant to act.
"""

import math

import torch


def solve_log_c(teacher_log_probs: torch.Tensor, log_floor: torch.Tensor, beta: float, iters: int = 40):
    """Solve ``sum_v max(c*pi_T(v), floor(v)) = 1`` for c, in log space.

    The sum is continuous and strictly increasing in c, so a bisection converges.
    The bracket is analytic rather than a guess:

    * lower: ``1 = sum max(c*pi_T, beta*pi_A) <= c + beta`` gives ``c >= 1-beta``.
    * upper: at ``c = 1/sum(pi_T)`` the teacher term alone already sums to 1, so
      the max-sum is at least 1. On a top-k support ``sum(pi_T) < 1``, which is
      why this bound can exceed 1 and a hard ``c <= 1`` would be wrong.

    Args:
        teacher_log_probs: (..., K) log pi_T over the evaluated support.
        log_floor: (..., K) log of the floor, i.e. ``log beta + log pi_A``.
        beta: floor strength in (0, 1).

    Returns:
        (..., 1) log c.
    """
    lo = torch.full_like(teacher_log_probs[..., :1], math.log1p(-beta))
    hi = -torch.logsumexp(teacher_log_probs, dim=-1, keepdim=True)
    for _ in range(iters):
        mid = (lo + hi) / 2
        over = torch.logsumexp(torch.maximum(mid + teacher_log_probs, log_floor), dim=-1, keepdim=True) > 0.0
        hi = torch.where(over, mid, hi)
        lo = torch.where(over, lo, mid)
    return (lo + hi) / 2


def relative_floor_target(teacher_log_probs: torch.Tensor, anchor_log_probs: torch.Tensor, beta: float):
    """Build the projection target ``q* = argmin_q KL(q || pi_T)  s.t.  q >= beta*pi_A``.

    Its closed form is ``q*(v) = max(c*pi_T(v), beta*pi_A(v))``: candidates the
    floor does not reach keep the teacher's relative preferences, and the rest
    are lifted to exactly the floor.

    Args:
        teacher_log_probs: (..., K) log pi_T on the support.
        anchor_log_probs: (..., K) log pi_A on the same support.
        beta: floor strength in (0, 1).

    Returns:
        target_log_probs: (..., K) log q*, normalized over the support.
        floor_binding: (..., K) bool, True where the floor is what sets q*.
        log_c: (..., 1) the solved normalizer.
    """
    log_floor = anchor_log_probs + math.log(beta)
    log_c = solve_log_c(teacher_log_probs, log_floor, beta)
    scaled_teacher = log_c + teacher_log_probs
    floor_binding = log_floor > scaled_teacher
    target_log_probs = torch.maximum(scaled_teacher, log_floor)
    # The solve already balances the support to 1; this only absorbs bisection
    # residue so the target is a distribution to float precision.
    target_log_probs = target_log_probs - torch.logsumexp(target_log_probs, dim=-1, keepdim=True)
    return target_log_probs, floor_binding, log_c
