"""compute_group_metrics 검증: 그룹을 학습 신호 유무로 나누는 로직.

실행: conda activate ICLR-verl && python -m pytest test_group_metrics.py -q -s
"""
import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.ppo.metric_utils import GROUP_SUCCESS_COUNTS_KEY, compute_group_metrics


def make_batch(groups):
    """groups: [(uid, [성공여부...]), ...] -> DataProto"""
    scores, uids = [], []
    for uid, outcomes in groups:
        for ok in outcomes:
            scores.append([float(ok)])
            uids.append(uid)
    return DataProto(
        batch=TensorDict({"token_level_scores": torch.tensor(scores)}, batch_size=len(scores)),
        non_tensor_batch={"uid": np.array(uids, dtype=object)},
    )


def test_three_shares_sum_to_one():
    m = compute_group_metrics(make_batch([
        ("a", [0, 0, 0, 0]),   # all fail
        ("b", [1, 1, 1, 1]),   # all success
        ("c", [1, 0, 0, 0]),   # informative
        ("d", [1, 1, 0, 0]),   # informative
    ]))
    print(f"\n  all_fail={m['training/groups/all_fail']:.2f} "
          f"all_success={m['training/groups/all_success']:.2f} "
          f"informative={m['training/groups/informative']:.2f}")
    assert m["training/groups/all_fail"] == 0.25
    assert m["training/groups/all_success"] == 0.25
    assert m["training/groups/informative"] == 0.5
    total = sum(m[f"training/groups/{k}"] for k in ("all_fail", "all_success", "informative"))
    assert abs(total - 1.0) < 1e-9, "세 비율의 합이 1이어야 한다"


def test_histogram_counts_groups_by_successes():
    m = compute_group_metrics(make_batch([
        ("a", [0, 0, 0, 0]),
        ("b", [0, 0, 0, 0]),
        ("c", [1, 0, 0, 0]),
        ("d", [1, 1, 0, 0]),
        ("e", [1, 1, 1, 1]),
    ]))
    hist = m[GROUP_SUCCESS_COUNTS_KEY]
    print(f"\n  히스토그램 {{성공 수: 그룹 수}} = {dict(sorted(hist.items()))}")
    assert hist == {0: 2, 1: 1, 2: 1, 4: 1}
    # 세 비율이 히스토그램에서 파생됨을 확인
    n = sum(hist.values())
    assert m["training/groups/all_fail"] == hist[0] / n
    assert m["training/groups/all_success"] == hist[4] / n


def test_all_fail_and_all_success_are_distinguished():
    """둘 다 advantage 0을 만들지만 원인이 반대다."""
    fail = compute_group_metrics(make_batch([("a", [0, 0, 0, 0])]))
    succ = compute_group_metrics(make_batch([("a", [1, 1, 1, 1])]))
    print(f"\n  전부 실패: all_fail={fail['training/groups/all_fail']:.0f} "
          f"all_success={fail['training/groups/all_success']:.0f}")
    print(f"  전부 성공: all_fail={succ['training/groups/all_fail']:.0f} "
          f"all_success={succ['training/groups/all_success']:.0f}")
    assert fail["training/groups/all_fail"] == 1.0 and fail["training/groups/all_success"] == 0.0
    assert succ["training/groups/all_success"] == 1.0 and succ["training/groups/all_fail"] == 0.0


def test_single_rollout_group():
    """G=1 이면 한 그룹은 전부 실패이거나 전부 성공이다 (informative 불가)."""
    m = compute_group_metrics(make_batch([("a", [1]), ("b", [0])]))
    print(f"\n  G=1: all_fail={m['training/groups/all_fail']:.1f} "
          f"all_success={m['training/groups/all_success']:.1f} "
          f"informative={m['training/groups/informative']:.1f}")
    assert m["training/groups/informative"] == 0.0


def test_realistic_grpo_batch():
    """G=16, 문제 8개 — 스모크 테스트와 같은 규모."""
    rng = np.random.default_rng(0)
    groups = [(f"p{i}", (rng.random(16) < p).astype(int).tolist())
              for i, p in enumerate([0.0, 0.0, 0.06, 0.1, 0.3, 0.5, 0.9, 1.0])]
    m = compute_group_metrics(make_batch(groups))
    hist = m[GROUP_SUCCESS_COUNTS_KEY]
    print(f"\n  G=16, 그룹 8개: all_fail={m['training/groups/all_fail']:.3f} "
          f"all_success={m['training/groups/all_success']:.3f} "
          f"informative={m['training/groups/informative']:.3f}")
    print(f"  히스토그램 = {dict(sorted(hist.items()))}")
    assert sum(hist.values()) == 8
    assert abs(sum(m[f"training/groups/{k}"] for k in
                   ("all_fail", "all_success", "informative")) - 1.0) < 1e-9
