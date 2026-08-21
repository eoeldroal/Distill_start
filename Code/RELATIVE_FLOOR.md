# relative-floor projection 구현 (2026-08-21)

우리 방법을 verl에 넣은 기록. 정의의 출처는 `Document/Draft.md` §5.1이고
수치 기준은 `Document/Cal_Beta_Before_train.md`와 `Document/toy_sims/floor_vs_kl.py`다.

## 무엇을 만들었나

verl에는 이미 on-policy distillation(OPD)이 있고, 그중 GKD 변형의 손실이
forward KL `Σ_v ν(v)·log(ν(v)/π_θ(v))`다. 여기서 target ν가 teacher인데,
우리는 그것을 relative-floor projection의 해로 바꿨다.

    q*(v) = max(c·π_T(v), β·π_A(v))

teacher를 최대한 따라가되, frozen anchor(Base)가 주던 확률의 β배 아래로는
어떤 candidate도 떨어뜨리지 않는다. c는 전체 합이 1이 되게 하는 공통 상수다.

`loss_mode="relative_floor_topk"` 로 선택한다. 기존 `forward_kl_topk` 경로는
건드리지 않았고(회귀 테스트 6개 통과), 새 함수를 등록하는 방식으로 얹었다.

## 바꾼 파일 (14개 파일 +725줄 수정 + 신규 projection.py 82줄)

**`verl/trainer/distillation/projection.py`** (신규 82줄) — 순수 수학.
engine에 종속되지 않는 부분을 분리해 fsdp와 (나중의) megatron 커널이 하나를 공유한다.
- `solve_log_c(...)`: 정규화 상수 c를 log 공간 이분법으로 푼다. 전부 log 공간인 이유는
  teacher가 짓누른 token의 확률이 float32에서 underflow하고 floor가 작동하는 자리가
  바로 그곳이기 때문이다. 탐색 구간은 추측이 아니라 해석적으로 온다: 아래는
  `1 = Σ max(c·π_T, β·π_A) ≤ c + β` 에서 c ≥ 1−β, 위는 `c = 1/Σπ_T`
  (그 값에서 teacher 항만으로 합이 1이 된다). top-k 지지집합에서는 Σπ_T < 1이라
  이 상한이 1을 넘을 수 있으므로 `c ≤ 1`로 자르면 틀린다.
- `relative_floor_target(...)`: q*를 만들고 어느 token에서 floor가 걸렸는지 함께 반환한다.

**`verl/trainer/distillation/fsdp/losses.py`** (+116) — FSDP 커널.
`compute_relative_floor_topk(...)`가 verl 형식(nested top-k)을 받아 손실을 계산한다.
지지집합은 **teacher top-k ∪ anchor top-k**다. floor는 anchor가 실제로 mass를 준
곳에서만 의미 있게 걸리므로 anchor의 top-k가 보호 대상을 덮고, teacher의 top-k는
teacher가 지키려는 것을 덮는다. 자기 top-k 밖의 값은 `NEG_LOG_PROB`로 두는데,
그 자리에서 floor가 걸리는 것이 정확한 동작이다(teacher가 짓누른 token의 q*는
`β·π_A`이고 teacher 값에 의존하지 않는다). 두 index 집합의 짝짓기는 정렬 후
`searchsorted`로 한다 — 모든 쌍을 비교하면 `[B, T, topk, topk]` 텐서가 필요하고
훈련 규모에서 그것은 student logits보다 크다.

**`verl/trainer/distillation/losses.py`** (+90) — 등록과 dispatch.
- `compute_topk_loss`가 `loss_mode`에 따라 커널을 고르고, relative-floor일 때
  `data["anchor_logprobs"]`/`["anchor_ids"]`를 함께 넘긴다. 없으면 명확한 KeyError.
- `@register_distillation_loss(... names=["relative_floor_topk"] ...)` 로 등록한
  함수가 손실을 unpad하고 진단 지표를 만든다.
- megatron 전략은 명시적으로 NotImplementedError (그 커널은 vocab-parallel 샤드
  위에서 손으로 쓴 backward를 쓰므로 target 교체에 별도 작업이 필요하다).

**`verl/workers/config/distillation.py`** (+22) — 설정.
- `relative_floor_beta: Optional[float] = None`
- 검증: relative_floor_topk인데 β가 없으면 오류, β가 (0,1) 밖이면 오류,
  `use_policy_gradient=True`면 오류(분포 신호가 낭비되므로).

## 새 진단 지표

| 지표 | 뜻 |
|---|---|
| `distillation/anchor_mass` | 지지집합이 담은 anchor 확률 (top-k 절단의 크기) |
| `distillation/floor_binding_count` | 위치당 floor가 걸린 token 수 (= k_bind의 실측) |
| `distillation/target_floor_mass` | q* 중 floor가 든 mass의 비율 |

`teacher_mass`, `student_mass`는 기존 경로와 같은 의미다.

## 검증

### 단위 테스트 36개 (`tests/test_relative_floor.py`) — 전부 통과

문서의 정의와 성질을 불변식으로 삼았다.

**정의 일치**
- toy 재현: Cost(0.4) = **0.0995** (문서 기준 0.0995), clamp된 token **2개**,
  c = **0.9293** — `toy_sims/floor_vs_kl.py`의 값과 소수점 넷째 자리까지 일치
- odds 보존: clamp 안 된 A:B odds가 teacher 6.0714 → q* 6.0714 (§5.1의 성질)
- 독립 참조 구현(확률 공간 이분법)과 최대 오차 **1.5e-07**

**수학적 성질** (β ∈ {0.1, 0.2, 0.4, 0.8} 전부)
- Σq* = 1: 오차 ≤ 2.4e-07
- floor 제약 `q*(v) ≥ β·π_A(v)`: 위반 0건
- teacher 하한 `q*(v) ≥ (1−β)·π_T(v)`: 위반 0건, c 최소값이 정확히 1−β 이상
  (β=0.4에서 c ≥ 0.6001) — `Cal_Beta_Before_train.md` §8의 부등식
- 최적성: 같은 floor 보장에서 KL(q*‖π_T) ≤ KL(mixture‖π_T) 전 β에서 성립
- 비용의 β 단조 증가
- β→0 극한: β를 10배 줄이면 비용도 10배 이상 줄어 π_T로 수렴
- 형식 유지 법칙: teacher가 한 token에 확률 1을 몰고 anchor가 그것을 무시하면
  q*가 그 token에 남기는 확률이 정확히 **1−β** (β=0.4 → 0.6000)

**verl 통합**
- 커널이 packed 형식 `(1, total_nnz, V)`을 받아 gradient가 student logits까지 흐른다
- **K=V일 때 전체 vocabulary 계산과 최대 오차 1.4e-06** — top-k 경로가 절단 없는
  정확한 계산과 같음을 보인다
- teacher/anchor top-k가 많이 겹치는 설정에서도 손실이 유한

**설정 검증** — β 누락/범위 이탈/policy-gradient 조합이 모두 막힌다

### 실제 모델 검증 (`tests/test_relative_floor_real.py`) — 통과

teacher Qwen3-4B, anchor Qwen3-1.7B-Base, anchor rollout에서 뽑은 state 15개
(plain 프롬프트, 위치 0/10/30/60/100).

| β | Cost(β) | clamp된 token 수 | floor mass | c 최소 | floor 위반 |
|---|---|---|---|---|---|
| 0.1 | 0.0097 | 45,641 | 0.011 | 0.969 | 0 |
| 0.2 | 0.0326 | 55,276 | 0.037 | 0.926 | 0 |
| 0.4 | **0.1115** | 65,950 | 0.124 | 0.719 | 0 |
| 0.8 | 0.3921 | 79,222 | 0.314 | 0.256 | 0 |

Cost(0.4) = 0.1115는 사전 분석의 실측 0.114와 사실상 같다 (teacher가 14B가 아닌
4B이고 state가 15개짜리 표본인데도 같은 자릿수).

top-k 절단의 실제 영향 (β=0.4):

| K | teacher mass | anchor mass | 지켜진 floor | Cost(전체) | Cost(top-k) |
|---|---|---|---|---|---|
| 64 | 0.9988 | 0.9837 | 98.5% | 0.1115 | 0.0958 |
| 128 | 0.9993 | 0.9901 | 99.1% | 0.1115 | 0.1014 |
| 512 | 0.9997 | 0.9969 | 99.7% | 0.1115 | 0.1083 |

K=128이면 floor 약속의 99.1%가 지켜지고 K=512면 99.7%다. 앞선 검토(top-512로
충분하다)와 일치한다.

### 회귀와 규약
- 기존 verl 테스트 `tests/workers/test_distillation_topk_symmetry_on_cpu.py` 6개 통과
  (`test_megatron_distillation_only_on_cpu.py`는 megatron 미설치로 수집 실패하는데,
  우리 변경 이전에도 동일하다)
- `ruff check` / `ruff format` 통과 (verl의 pre-commit 규칙)

### 성능 (N=16384, K=128, V=151936 — 실제 훈련 micro-batch 규모)

| | 커널이 추가로 쓰는 메모리 | 시간 |
|---|---|---|
| verl 원본 `compute_forward_kl_topk` | +4.94 GB | 0.21 s |
| 우리 `compute_relative_floor_topk` | **+4.85 GB** | **0.13 s** |

anchor 쪽 분포를 하나 더 다루는데도 원본보다 조금 낮은데, 원본이 진단용으로 만드는
`[B, T, topk, topk]` 겹침 텐서(K=128에서 0.25 GB)를 우리는 만들지 않기 때문이다.
두 커널 모두 `[B, T, V]` log_softmax 버퍼(4.64 GB)가 지배한다.

## 로깅

문서 §3.6 의 기제 사슬(floor 작동 → 진입 확률 → E_j → D_succ → all-fail 감소 →
informative 증가)의 고리이거나 그 해석에 필요한 통제 변수만 남겼다. 얼린 checkpoint 에서
사후 계산되는 값(E_j, V_j, B, D_succ)은 로깅하지 않는다.

**스칼라 (wandb 그래프로 바로 뜬다)**

| 지표 | 뜻 |
|---|---|
| `distillation/loss` | Cost(β) 의 훈련 중 실현값 |
| `distillation/teacher_mass`, `anchor_mass` | 지지집합이 담은 각 모델의 확률 (top-k 절단의 크기) |
| `distillation/student_mass` | 같은 지지집합에서 student 의 mass |
| `distillation/floor_binding_count` (+max) | 위치당 floor 가 이긴 token 수 = k_bind 의 실측 |
| `distillation/target_floor_mass` (+max) | q* 중 floor 가 든 mass. 이론 상한이 β 다 |
| `distillation/floor_binding/pos{0,1,2-3,4-7,8-15,16plus}` | 위 개수의 위치별 분해 |
| `training/groups/{all_fail,all_success,informative}` | GRPO 그룹을 학습 신호 유무로 나눈 비율 |

위치별 분해가 필요한 이유: 평균 하나로는 floor 가 앞쪽에 몰려 작동하는지 알 수 없는데,
문서는 창 앞부분의 binding 을 진입 보존(k_bind), 뒷부분을 실행 보존(V) 의 진단으로 나눈다.
구간은 사전 분석이 쓴 것을 그대로 상속해 두 프로파일을 직접 대조할 수 있게 했다.
그리고 anchor 는 고정이지만 student 가 방문하는 state 는 훈련 중 변하므로, 이 프로파일은
사후에 복원되지 않는다.

실측된 모양(스모크, β=0.4, K=64 → 지지집합 128칸):

| 위치 | 걸린 token 수 |
|---|---|
| pos0 | 64.0 |
| pos1 | 49.9 |
| pos2-3 | 47.2 |
| pos4-7 | 38.1 |
| pos8-15 | 32.8 |
| pos16+ | 24.5 |

앞쪽에 몰리는 모양이 사전 분석과 일치한다. pos0 이 64.0 인 것은 지지집합의 teacher 쪽
절반이 전부 걸렸다는 뜻으로, teacher 가 첫 token 에 확률을 몰아주고 anchor 가 그것을
모를 때 나타나는 패턴이다. `target_floor_mass_max` 가 0.39998 로 관측된 것도 이론값 β 에
도달한 사례다.

**표 (필요할 때 열어 보는 원자료)**

`training/groups/success_counts` — `{그룹 내 성공 수: 그룹 개수}` 히스토그램을 매 step 한
줄. DAPO 의 `filtered_reward_counts` 패턴을 일반화한 `StepHistogramTableLogger` 로 넣었고,
verl 의 `Counter` 병합 경로를 타서 iteration 들이 합산된다. 스칼라 3개가 이 표에서 파생되며,
그룹 내 성공률 분포도 여기서 정확히 나온다(advantage max 로 역산할 필요가 없다).

세 비율 중 `informative = 1 − all_fail − all_success` 이지만 셋 다 남겼다. 계산 비용이 0 이고
informative 가 논문 그림의 축이며, 셋을 나란히 봐야 informative 의 증가가 all_fail 감소에서
온 것인지 구분되기 때문이다.

## anchor 를 어디에 두는가 — 두 경로

anchor 는 매 step student 의 문장을 채점해야 하므로 살아 있어야 한다. 두 방법이 있고
`loss_mode=relative_floor_topk` 는 둘 중 하나를 요구한다.

### (A) `distillation.anchor_model` — 전용 추론 서버

teacher 와 똑같이 sglang 서버로 올린다. 같은 설정 타입, 같은 서버 기동 코드, 같은 채점
함수를 쓰고 teacher pool 을 나눠 쓴다. teacher 가 아니므로 routing 대상이 아니고, 모든
예제가 teacher 와 anchor 양쪽에서 채점된다.

대가는 **GPU 1장**이다. anchor 가 student 와 같은 1.7B 라 3.4 GB 만 쓰는데 80 GB 카드를
통째로 차지한다.

### (B) `distillation.anchor_from_ref=True` — reference policy 에서 채점 (권장)

verl 의 PPO reference policy 는 정의상 우리 anchor 와 같은 물건이다: student 의 훈련 시작
가중치를 얼려 둔 같은 크기의 모델(`ref_config.model_config = deepcopy(model_config)`).
그리고 그것은 actor 와 **같은 GPU, 같은 worker 프로세스**에 산다.

그래서 anchor 를 별도로 올리지 않고 ref 의 forward 에서 top-k 를 뽑는다. `infer_batch`
가 `loss_function` 을 받는 구조가 이미 있어서, ref 의 loss_fn 에 `compute_topk_scores`
를 넣으면 logits processor 자리에서 자기 top-k 를 내보낸다.

**추가 GPU 0장.** `actor_rollout_ref.ref.fsdp_config.param_offload=True` 로 가중치를 CPU
에 파킹하면 상주 메모리도 아낀다.

### 두 경로가 같은 결과를 낸다

`tests/test_anchor_equivalence_e2e.py` — 실제 모델(teacher Qwen3-4B, anchor
Qwen3-1.7B-Base), 고정된 44 위치 문장:

| 지표 | (A) 서버 경로 | (B) 엔진 경로 | 차이 |
|---|---|---|---|
| distillation_losses | 0.360319 | 0.360319 | **0** |
| student_mass | 0.982295 | 0.982295 | **0** |
| teacher_mass | 0.988448 | 0.988448 | **0** |
| anchor_mass | 0.980608 | 0.980608 | **0** |
| floor_binding_count | 55.045456 | 55.045456 | **0** |
| target_floor_mass | 0.098486 | 0.098486 | **0** |

top-k 값과 id 도 완전 일치한다. 서버가 하는 연산(`log_softmax(logits).topk(K)`)과 엔진이
하는 연산이 같기 때문이다.

**훈련 로그로는 이 비교가 성립하지 않는다.** verl 의 `rollout.do_sample` 은
`sampling_params` 로 전달되지 않고(REMAX 전용 per-sample override 만 있다), `temperature=0`
은 forward 의 `temperature.clamp(min=1e-8)` 때문에 logits 를 10^8 배 증폭해 분포를 깨뜨린다
(`mass > 1` 로 관측된다). 그래서 두 실행의 rollout 이 달라지고 지표가 갈린다. 동등성은
문장을 고정해 확인해야 한다.

### anchor 의 offload 는 verl 이 이미 강제한다

`actor_rollout_ref.ref.fsdp_config.param_offload` 는 anchor 에 대해 **아무 효과가 없다.**
ref 블록에는 `forward_only: true` 가 기본으로 들어 있고, FSDP 초기화가 그것을 보고
`CPUOffload(offload_params=True)` 를 강제한다.

```python
# transformer_impl.py:413 — FSDP1 초기화
# We force reference policy to use CPUOffload to save memory.
if self.engine_config.forward_only:
    cpu_offload = CPUOffload(offload_params=True)
    self._is_offload_param = False       # 수동 offload 스위치를 끈다
    self._is_offload_optimizer = False

# transformer_impl.py:827 — to() 는 조기 반환
if self.engine_config.forward_only:
    # force cpu_offload
    return
```

즉 offload 는 컨텍스트 진입/이탈 시 우리가 옮기는 것이 아니라, **FSDP 가 layer 단위로
필요할 때만 GPU에 올리고 곧바로 내리는 방식**이다. 시점이 "3단계 전체"가 아니라 "각 layer
의 forward 순간"이라 훨씬 촘촘하다.

실측이 이를 확인한다 (step 2~3 평균, GPU 4/5/6):

| | offload=True | offload=False |
|---|---|---|
| anchor 채점 시간 | 0.523 s | 0.497 s |
| actor 업데이트 | 0.572 s | 0.571 s |
| step 시간 | 7.961 s | 8.795 s |
| **actor peak mem** | **18.801 GB** | **18.801 GB** |

peak 메모리가 소수점까지 동일하다. 두 실행의 config 는 실제로 달랐고(`param_offload:
True` vs `False`) `forward_only: True` 는 양쪽 모두였다. 손실 차이(0.548 vs 0.499)는 rollout
이 확률적이라 문장이 달라진 것이며 offload 와 무관하다.

**따라서 이 인자는 건드릴 필요가 없다.** anchor 는 항상 CPU 상주 + layer 단위 로딩이고,
파이프라인 단계 분리는 verl 이 이미 보장한다.

### 실측 리소스 (스모크 규모, student 1.7B / teacher 4B)

| | GPU | sglang 서버 | step 시간 | actor peak mem |
|---|---|---|---|---|
| (A) anchor_model | 4장 | 4개 | 8.7 s | 18.22 GB |
| (B) anchor_from_ref | **3장** | 3개 | 9.2 s | 18.80 GB |

GPU 1장이 사라지고, step 시간 +6%, actor 메모리 +0.6 GB 다. 실제 실험에서 teacher 가
14B TP=2 라면 (A) 는 pool 3장(teacher 2 + anchor 1), (B) 는 2장이 된다.

### 구현이 건드린 곳

- `verl/trainer/distillation/losses.py`: `compute_topk_scores` — 엔진이 loss_fn 을 두 번
  부른다(logits processor 로, 손실로). 첫 호출에서 top-k 를 내보내고, 두 번째는 gradient
  없는 forward 이므로 더미 손실을 돌려준다.
- `verl/workers/engine/fsdp/transformer_impl.py`: logits processor 출력 계약을 스칼라
  `(total_nnz,)` 에서 `(total_nnz, ...)` 로 넓혔다. gather/unpad/nest 경로가 token 차원만
  다루므로 3D 도 그대로 통과한다(`nested_tensor_from_jagged` 는 3D 를 지원한다).
- `verl/workers/engine_workers.py`: `anchor_from_ref` 면 ref 에 `compute_topk_scores` 를
  붙이고, `compute_anchor_topk` RPC 를 추가했다(`compute_ref_log_prob` 와 같은 형태).
- `verl/trainer/ppo/v1/trainer_base.py`: fit 루프에 `_compute_anchor_topk` 를 넣었다.
  anchor 필드는 teacher 필드와 같은 시퀀스 전체 폭이어야 한다(response 만 자르면 커널의
  shape assert 에 걸린다).
- `verl/trainer/ppo/utils.py`: `need_reference_policy` 가 `anchor_from_ref` 도 참으로
  본다. KL 항이 필요 없어도 ref 가 만들어져야 하기 때문이다.

## 알려진 한계

- **megatron 전략 미지원.** 그 커널은 vocab-parallel 샤드 위에서 손으로 쓴 backward를
  쓰므로 target 교체에 별도 작업이 필요하다. 명시적 `NotImplementedError`로 막았다.
- **veomni의 fused kernel 경로 미지원.** `transformer_impl.py:1348`의 고정 키 목록이
  teacher 전용 세 필드만 꺼내므로, 그 경로가 켜지면 anchor 진단이 빠져 KeyError가 난다.
  우리가 쓰는 것은 그 아래의 일반 logit-processor 경로(1379행)다.

## 실행

```bash
conda activate ICLR-verl
cd Code/tests
CUDA_VISIBLE_DEVICES=3 python -m pytest test_relative_floor.py -q -s   # 단위 36개
CUDA_VISIBLE_DEVICES=3 python test_relative_floor_real.py              # 실제 모델
```

훈련 설정 (권장: anchor_from_ref):

```
distillation.enabled=True
distillation.anchor_from_ref=True
distillation.teacher_models.teacher_model.model_path=<teacher>
distillation.distillation_loss.loss_mode=relative_floor_topk
distillation.distillation_loss.relative_floor_beta=0.4
distillation.distillation_loss.topk=128
distillation.distillation_loss.use_policy_gradient=False
distillation.distillation_loss.use_task_rewards=False
actor_rollout_ref.ref.fsdp_config.param_offload=True
```

전용 서버 방식은 `anchor_from_ref` 대신
`distillation.anchor_model.model_path=<anchor>` 를 준다(둘은 배타적이다).

재현 스크립트: `verl_smoke/run_floor_ref_smoke.sh` (anchor_from_ref),
`verl_smoke/run_floor_smoke.sh` (anchor_model).
