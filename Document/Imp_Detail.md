# 구현 상세: relative-floor projection과 학습 파이프라인

이 문서는 §5.1의 relative-floor projection을 verl에서 어떻게 계산하는지, 그리고 그 계산이
학습 파이프라인의 자원을 어떻게 쓰는지를 적는다. 방법의 정의와 β의 유도는 Draft.md §5.1과
Cal_Beta_Before_train.md에 있고, 여기서는 그 정의가 실제 텐서 연산과 GPU 배치로 내려오는
과정을 다룬다. 코드는 `Code/verl`(verl 0.10.0.dev)에 있다.

## 1. 세 모델

계산에 세 분포가 등장하고 역할이 서로 다르다.

**teacher π_T**는 배워야 할 실력의 원천이다. 별도 GPU의 추론 서버에 얼려 두고 생성은 시키지
않는다. student가 만든 문장을 받아 각 위치에서 자기라면 어떤 token에 얼마의 확률을 줬을지
채점한다.

**anchor π_A**는 보호할 대상의 기준이다. 역시 얼려 두고 채점만 하며, 훈련 내내 변하지 않는다.
anchor는 student의 훈련 시작 가중치, 즉 Base다.

**student π_θ**는 유일하게 학습되는 모델이다. 문장을 생성하는 것도 student이므로 채점은 항상
student가 실제로 방문한 state에서 이루어진다(on-policy distillation). 훈련이 진행되면 anchor는
고정된 채 student만 움직이므로 둘의 거리가 벌어진다.

## 2. 손실

한 위치에서 숫자가 어떻게 흐르는지를 실제 코드로 추적한 결과다(`Code/tests/trace_loss.py`).
vocabulary를 8개, top-k를 4개로 줄인 설명용 예시이며 β=0.4이다.

### 2.1 계산 대상을 정한다

teacher와 anchor는 각자 top-4만 돌려준다.

    teacher top-4 : Let, We, First, Note
    anchor  top-4 : First, Note, Since, Observe

둘을 이어 붙여 지지집합을 만든다. 앞 4칸이 teacher의 것, 뒤 4칸이 anchor의 것이고, 양쪽에
모두 있는 token(First, Note)은 뒤쪽 사본을 버린다. 같은 token을 두 번 세면 target이 분포가
아니게 된다. 남는 대상은 6개다.

겹침을 찾는 방법이 성능에 걸린다. 모든 쌍을 비교하면 `[B, T, topk, topk]` 텐서가 필요하고,
실제 훈련 규모(위치 16,384개, topk 128)에서 그것은 1 GB로 student logits보다 크다. 그래서
teacher의 id를 정렬해 두고 anchor의 id를 이진 탐색한다(`searchsorted`). 비용이 O(K log K)로
내려가고 K² 텐서가 사라진다.

### 2.2 두 모델의 값을 나란히 놓는다

| token | π_T | π_A | floor = 0.4·π_A |
|---|---|---|---|
| Let | 0.7000 | 0 (모름) | 0 |
| We | 0.2000 | 0 (모름) | 0 |
| First | 0.0600 | 0.3000 | 0.1200 |
| Note | 0.0300 | 0.2500 | 0.1000 |
| Since | 0 (모름) | 0.1500 | 0.0600 |
| Observe | 0 (모름) | 0.1000 | 0.0400 |

자기 top-k 밖의 값을 0으로 두는 것이 이 표의 판단이다. 두 방향 모두 옳은 동작이 된다.

Let은 anchor의 top-4에 없으니 anchor 확률을 모른다. 그러나 top-4 밖이라는 사실 자체가 그
확률이 작다는 뜻이므로 floor도 0이 되어 제약이 걸리지 않는다. anchor가 아끼지 않았던 token은
보호할 이유가 없다.

Since는 teacher의 top-4에 없으니 teacher 확률을 모른다. 0으로 두면 floor가 무조건 이기는데,
이것도 정확하다. teacher가 짓눌러 버린 token의 q*는 β·π_A로 정해지고 teacher 값이 식에 아예
나타나지 않기 때문이다. 몰라도 답이 틀리지 않는다.

### 2.3 정규화 상수를 구한다

floor 때문에 일부 token이 올라가면 합이 1을 넘는다. teacher 쪽을 공통 비율 c로 줄여 균형을
맞춰야 하고, c는 Σ max(c·π_T, β·π_A) = 1을 만족하는 값이다. 합이 c에 대해 연속이고 단조
증가하므로 이분법이 수렴한다.

탐색 구간은 추측이 아니라 해석적으로 온다. 아래는 1 = Σ max(c·π_T, β·π_A) ≤ c + β에서
c ≥ 1−β = 0.6이고, 위는 c = 1/Σπ_T = 1.0101이다(그 값에서 teacher 항만으로 합이 1이 되므로
max의 합은 1 이상이다). top-k 지지집합에서는 Σπ_T < 1이라 이 상한이 1을 넘을 수 있고, 따라서
c ≤ 1로 자르면 틀린다. 이 예시에서 찾은 값은 c = 0.7556이다.

계산은 전부 log 공간에서 한다. teacher가 짓누른 token의 확률은 float32에서 underflow하고,
floor가 작동하는 자리가 바로 그곳이다.

### 2.4 target을 만든다

각 token에서 c·π_T와 β·π_A 중 큰 쪽이 이긴다.

| token | c·π_T | β·π_A | 승자 | q* | teacher 대비 |
|---|---|---|---|---|---|
| Let | 0.5289 | 0 | teacher | 0.5289 | 0.76배 |
| We | 0.1511 | 0 | teacher | 0.1511 | 0.76배 |
| First | 0.0453 | 0.1200 | floor | 0.1200 | 2.00배 |
| Note | 0.0227 | 0.1000 | floor | 0.1000 | 3.33배 |
| Since | 0 | 0.0600 | floor | 0.0600 | 7.50배 |
| Observe | 0 | 0.0400 | floor | 0.0400 | 40.0배 |

Σq* = 1.000000으로 분포가 성립한다. 표에서 두 가지가 읽힌다.

teacher가 이긴 token들은 서로의 비율이 그대로다. Let과 We가 둘 다 0.76배로 줄었으므로 teacher가
Let을 We보다 3.5배 선호했다면 q*에서도 정확히 3.5배다. §5.1의 odds 보존이 이것이다. teacher의
판단을 왜곡하지 않고 총량만 조금 내준다.

floor가 이긴 token들은 anchor가 준 확률에 비례한다. anchor가 많이 아꼈던 First는 0.12, 적게
아꼈던 Observe는 0.04다. 보호의 크기가 token마다 anchor의 판단을 따라간다.

### 2.5 손실을 계산한다

forward KL이므로 Σ q*(v)·[log q*(v) − log π_θ(v)] 형태다.

| token | q* | π_θ | log(q*/π_θ) | 기여도 |
|---|---|---|---|---|
| Let | 0.5289 | 0.3984 | +0.2833 | +0.1498 |
| We | 0.1511 | 0.2490 | −0.4995 | −0.0755 |
| First | 0.1200 | 0.1494 | −0.2192 | −0.0263 |
| Note | 0.1000 | 0.0996 | +0.0040 | +0.0004 |
| Since | 0.0600 | 0.0598 | +0.0040 | +0.0002 |
| Observe | 0.0400 | 0.0299 | +0.2917 | +0.0117 |
| | | | | 합 0.0604 nats |

부호가 양수면 student가 q*보다 적게 주고 있어 올려야 하고(Let, Observe), 음수면 많이 주고
있어 내려야 한다(We, First).

가중치가 q*라는 점이 방향을 정한다. q*가 큰 token의 오차가 손실을 지배하므로, 이 손실은 q*가
확률을 준 곳을 student가 반드시 덮게 만든다(mass-covering). floor가 지켜 준 token을 student가
무시하지 못하게 하는 것이 목적이므로 필요한 방향이다.

### 2.6 vanilla distillation과의 차이

같은 자리에서 보통의 distillation은 target이 teacher 그 자체다. First의 target은 0.0600이고
우리는 0.1200을 쓴다. Observe는 vanilla가 0.0010, 우리가 0.0400으로 40배다.

차이는 한 줄로 요약된다. vanilla는 anchor를 보지 않고, 우리는 anchor가 아꼈던 token에 하한선을
둔다. 그 하한선의 높이가 β다.

## 3. 코드 배치

verl에는 on-policy distillation이 이미 있고 그중 GKD 변형의 손실이 forward KL
Σ_v ν(v)·log(ν(v)/π_θ(v))이다. 여기서 target ν가 teacher인데, 우리는 그것을 q*로 바꿨다. 새
알고리즘을 얹은 것이 아니라 등록된 손실을 하나 추가한 것이다.

| 파일 | 역할 |
|---|---|
| `trainer/distillation/projection.py` | q*를 만드는 순수 수학. engine에 종속되지 않아 fsdp와 (나중의) megatron이 공유한다 |
| `trainer/distillation/fsdp/losses.py` | FSDP 커널. 지지집합 구성, searchsorted 짝짓기, KL |
| `trainer/distillation/losses.py` | 손실 등록과 dispatch, anchor 추출, 위치별 진단 |
| `workers/config/distillation.py` | `relative_floor_beta`, `anchor_from_ref`, `anchor_model`과 그 검증 |

선택은 `loss_mode="relative_floor_topk"`이고, 기존 `forward_kl_topk` 경로는 건드리지 않았다.
megatron 전략은 명시적으로 막았다. 그 커널은 vocab-parallel 샤드 위에서 손으로 쓴 backward를
쓰므로 target 교체에 별도 작업이 필요하다.

커널의 비용은 실제 훈련 규모(위치 16,384개, vocabulary 151,936, topk 128)에서 메모리 +4.85 GB,
시간 0.13초다. verl 원본 `compute_forward_kl_topk`가 같은 조건에서 +4.94 GB, 0.21초이므로 anchor
분포를 하나 더 다루면서도 원본보다 낮다. 원본이 진단용으로 만드는 K² 겹침 텐서를 만들지 않기
때문이다. 두 커널 모두 `[B, T, V]` log_softmax 버퍼(4.64 GB)가 비용을 지배한다.

훈련 설정은 다음과 같다.

    distillation.enabled=True
    distillation.anchor_from_ref=True
    distillation.teacher_models.teacher_model.model_path=<teacher>
    distillation.distillation_loss.loss_mode=relative_floor_topk
    distillation.distillation_loss.relative_floor_beta=0.4
    distillation.distillation_loss.topk=128
    distillation.distillation_loss.use_policy_gradient=False
    distillation.distillation_loss.use_task_rewards=False

`use_policy_gradient=False`가 GKD 형태(분포를 직접 backprop)를 고르는 것이고,
`use_task_rewards=False`는 이 단계에서 task reward를 섞지 않는다는 뜻이다. 두 값 모두 config
검증이 강제한다. relative_floor_topk에 policy gradient를 붙이면 분포 신호가 낭비되므로 오류로
막았다.

## 4. anchor를 어디에 두는가

anchor는 매 step student의 문장을 채점해야 하므로 살아 있어야 한다. 방법이 둘이고 결과는 같다.

**(A) `anchor_model`.** teacher와 똑같이 sglang 서버로 올린다. 같은 설정 타입, 같은 서버 기동
코드, 같은 채점 함수를 쓰고 teacher pool을 나눠 쓴다. teacher가 아니므로 routing 대상이 아니고
모든 예제가 양쪽에서 채점된다. 대가는 GPU 한 장이다. anchor가 student와 같은 1.7B라 3.4 GB만
쓰는데 카드를 통째로 차지한다.

**(B) `anchor_from_ref=True`.** verl의 PPO reference policy는 정의상 우리 anchor와 같은
물건이다. student의 훈련 시작 가중치를 얼려 둔 같은 크기의 모델이고(`ref_config.model_config =
deepcopy(model_config)`), actor와 같은 GPU, 같은 worker 프로세스에 산다. 그래서 anchor를 따로
올리지 않고 ref의 forward에서 top-k를 뽑는다. `infer_batch`가 `loss_function`을 받는 구조가 이미
있어서 ref의 loss_fn에 추출 함수를 넣으면 logits processor 자리에서 자기 top-k를 내보낸다.
추가 GPU가 없다.

구현이 건드린 곳은 넷이다. `losses.py`의 `compute_topk_scores`는 엔진이 loss_fn을 두 번 부르는
것을 처리한다(logits processor로, 그리고 손실로). `transformer_impl.py`는 logits processor의
출력 계약을 스칼라 `(total_nnz,)`에서 `(total_nnz, ...)`로 넓혔다. gather/unpad/nest 경로가 token
차원만 다루고 `nested_tensor_from_jagged`가 3D를 지원하므로 top-k 텐서가 그대로 통과한다.
`engine_workers.py`는 `compute_anchor_topk` RPC를 추가했고, `ppo/utils.py`의
`need_reference_policy`가 `anchor_from_ref`도 참으로 본다. KL 항이 필요 없어도 ref가 만들어져야
하기 때문이다.

두 경로가 같은 결과를 낸다는 것은 실제 모델로 확인했다(`Code/tests/test_anchor_equivalence_e2e.py`,
teacher Qwen3-4B와 anchor Qwen3-1.7B-Base, 고정된 44 위치 문장).

| 지표 | (A) 서버 경로 | (B) 엔진 경로 | 차이 |
|---|---|---|---|
| distillation_losses | 0.360319 | 0.360319 | 0 |
| student_mass | 0.982295 | 0.982295 | 0 |
| teacher_mass | 0.988448 | 0.988448 | 0 |
| anchor_mass | 0.980608 | 0.980608 | 0 |
| floor_binding_count | 55.045456 | 55.045456 | 0 |
| target_floor_mass | 0.098486 | 0.098486 | 0 |

top-k 값과 id도 완전히 일치한다. 서버가 하는 연산(`log_softmax(logits).topk(K)`)과 엔진이 하는
연산이 같기 때문이다.

훈련 로그로는 이 비교가 성립하지 않는다. verl의 `rollout.do_sample`은 sampling_params로 전달되지
않고(REMAX 전용 per-sample override만 있다), `temperature=0`은 forward의 `clamp(min=1e-8)` 때문에
logits를 10⁸배 증폭해 분포를 깨뜨린다(mass > 1로 관측된다). 그래서 두 실행의 rollout이 갈리고
지표가 달라진다. 동등성은 문장을 고정해 확인해야 한다.

이하는 (B)를 전제로 한다.

## 5. 파이프라인과 메모리 시분할

GPU는 두 pool로 나뉜다.

    student pool (trainer.n_gpus_per_node)
      ① student   FSDP 샤딩, 학습 대상. 옵티마이저와 gradient를 가진다
      ② anchor    같은 프로세스의 두 번째 모델, 얼림. forward_only=True
      ③ rollout   sglang 서버, 같은 GPU에 colocate

    teacher pool (distillation.n_gpus_per_node)
      ④ teacher   sglang 서버, 얼림, 상주

vanilla OPD와 GPU 수가 같다. anchor가 학습 GPU에 얹혀 있고 상주하지도 않는다.

학습 GPU에서 세 모델이 시간축으로 자리를 나눈다. 괄호의 시간은 실측이다(스모크 규모, step 2~3
평균).

    1. 생성 (4.35 s)
       rollout   가중치 + KV 캐시 상주
       student   유휴
       anchor    CPU

    2. teacher 채점 (다른 GPU, 비동기)
       학습 GPU는 대기

    3. anchor 채점 (0.52 s)
       rollout   KV 캐시 해제
       anchor    layer 1 → GPU → 계산 → CPU
                 layer 2 → GPU → 계산 → CPU
                 ...  모델 전체가 동시에 올라오지 않는다
       student   유휴

    4. student 학습 (0.57 s)
       student   가중치 + 활성값 + gradient      ← peak 구간
       anchor    CPU (한 바이트도 없다)
       rollout   sleep

    5. 가중치 동기화 (2.13 s)
       student → rollout 서버로 새 가중치 전송. 전송 중 rollout은 sleep

anchor의 offload 시점이 특히 촘촘하다. `CPUOffload(offload_params=True)`는 3단계 시작에 3.4 GB를
올리고 끝에 내리는 방식이 아니라, 각 layer의 forward 순간에 그 layer만 올리고 즉시 내린다.
어느 순간에도 모델 전체가 GPU에 있지 않고, student가 peak를 쓰는 4단계에는 anchor가 GPU에 전혀
없다.

이 배치는 우리가 인자로 켠 것이 아니라 verl이 강제한다. ref 블록에는 `forward_only: true`가
기본으로 들어 있고, FSDP 초기화가 그것을 보고 offload를 못 박는다.

    # transformer_impl.py:411
    # We force reference policy to use CPUOffload to save memory.
    if self.engine_config.forward_only:
        cpu_offload = CPUOffload(offload_params=True)
        self._is_offload_param = False       # 수동 스위치를 끈다
        self._is_offload_optimizer = False

그리고 `to()`가 조기 반환해서 수동 이동조차 막는다.

    # transformer_impl.py:827
    if self.engine_config.forward_only:
        # force cpu_offload
        return

따라서 이 최적화는 설정이 아니라 불변식이다. `actor_rollout_ref.ref.fsdp_config.param_offload`를
어느 값으로 두어도 동작이 같고, 설정 실수로 학습 peak에 anchor가 끼어들 여지가 없다. 실측이
이를 확인한다.

| | param_offload=True | param_offload=False |
|---|---|---|
| anchor 채점 시간 | 0.523 s | 0.497 s |
| student 업데이트 | 0.572 s | 0.571 s |
| actor peak memory | 18.801 GB | 18.801 GB |

peak 메모리가 소수점까지 같다. 두 실행의 config는 실제로 달랐고(`param_offload: True` 대
`False`) `forward_only: True`는 양쪽 모두였다.

## 6. 비용

step당 단계별 시간이다. teacher 채점은 다른 GPU에서 비동기로 돌아 이 표에 나타나지 않는다.

| 단계 | 시간 | 비중 |
|---|---|---|
| 생성 | 4.35 s | 55% |
| **anchor 채점** | **0.52 s** | **6.6%** |
| old_log_prob | 0.30 s | 3.8% |
| advantage | 0.07 s | 0.9% |
| student 업데이트 | 0.57 s | 7.2% |
| 가중치 동기화 | 2.13 s | 27% |
| 합계 | 7.96 s | |

vanilla OPD와 나란히 놓으면 이렇다.

| | vanilla OPD | relative-floor |
|---|---|---|
| GPU 수 | student pool + teacher pool | 동일 |
| 학습 GPU 상주 메모리 | student + rollout | 동일 (anchor는 CPU) |
| 학습 peak 메모리 | 기준 | 약 +0.6 GB (anchor forward의 활성값과 top-k 텐서) |
| step 시간 | 기준 | +6.6% (anchor 채점) |
| RL 단계 비용 | 기준 | 동일 (anchor 불필요) |

anchor 가중치가 3.4 GB인데 peak 증가가 0.6 GB인 이유가 §5의 시분할이다. anchor는 student의 peak
구간에 GPU에 없고, 남는 0.6 GB는 anchor forward 자체의 활성값과 top-k 텐서다.

teacher가 커지면 상대 비용이 더 줄어든다. anchor는 student와 같은 크기이고 teacher 채점과 다른
GPU에서 병렬로 돌기 때문에, teacher가 14B가 되어 그쪽이 병목이 되면 anchor의 0.52초는 가려진다.

이 비용은 distillation 단계에만 발생하고, 그 뒤로는 0이다. 우리 방법이 하는 일은 더 나은
checkpoint를 만드는 것이고 RL에 넘긴 뒤에는 anchor가 필요 없다. §6의 prevention 대 repair 대비에서
이 성격이 갈린다. repair 계열(entropy-controlled RL, self-reheating, anchor를 다시 불러오는 복구)은
RL 단계에서 계속 비용을 내지만, prevention은 앞단에서 한 번 내고 끝낸다.

## 7. 로깅

§3.6의 기제 사슬(floor 작동 → \(E^{\mathrm{entry}}\) 증가 → \(E^{\mathrm{gen}}\) 회복 →
\(D_{\mathrm{succ}}\) 증가 → all-fail 감소 → informative 증가)의 고리이거나, 그 해석에 필요한
통제 변수만 남겼다. 얼린 checkpoint에서 사후에 계산하는 \(E^{\mathrm{entry}}\),
\(E^{\mathrm{gen}}\), \(V\), \(B\), \(D_{\mathrm{succ}}\)는 학습 로그에 넣지 않는다.

스칼라는 wandb에서 바로 시계열로 보이고 arm 간 비교도 자동이다.

| 지표 | 뜻 |
|---|---|
| `distillation/loss` | KL(q*‖π_θ). student가 target에서 얼마나 먼지 |
| `distillation/cost_beta` (+max) | Cost(β) = KL(q*‖π_T). floor가 target을 teacher에서 끌어낸 값 |
| `distillation/cost_beta/pos{0,1,2-3,4-7,8-15,16plus}` | 위 값의 위치별 분해 |
| `distillation/teacher_mass`, `anchor_mass`, `student_mass` | 지지집합이 담은 각 분포의 확률. top-k 절단의 크기 |
| `distillation/floor_binding_count` (+max) | 위치당 floor가 이긴 token 수. k_bind의 실측 |
| `distillation/target_floor_mass` (+max) | q* 중 floor가 든 mass. 이론 상한이 β다 |
| `distillation/floor_binding/pos{0,1,2-3,4-7,8-15,16plus}` | 위 개수의 위치별 분해 |
| `training/groups/{all_fail,all_success,informative}` | GRPO 그룹을 학습 신호 유무로 나눈 비율 |

loss와 cost_beta는 다른 값이다. 커널의 `kl_divergence(log_q, log_p)`가 `Σ p(log p − log q)`,
즉 KL(p‖q)이고 손실은 `log_p`에 target을 `log_q`에 student를 받으므로 KL(q*‖π_θ)다. β 결정
규칙 β* = min(β_knee, Cost⁻¹(δ))가 쓰는 값은 KL(q*‖π_T)이고 그것이 cost_beta다. 지지집합 밖에서
teacher의 log 확률이 NEG_LOG_PROB이면 KL이 발산하므로 cost_beta는 clamp된 teacher로 계산한다.
그래서 이 값은 하한이고 조임 정도를 log_prob_min_clamp가 정한다.

위치별 분해가 필요한 이유는 평균 하나로는 floor가 앞쪽에 몰려 작동하는지 알 수 없다는 것이다.
§1.6이 창 앞부분의 binding을 진입 보존(k_bind), 뒷부분을 branch 내부 실행(V)의 진단으로 나누므로
둘을 섞으면 retention 계산도 해석도 못 한다. 구간은 Cal_Beta_Before_train.md의 사전 분석이 쓴
것을 그대로 상속해 두 프로파일을 직접 대조할 수 있게 했다. 그리고 anchor는 고정이지만 student가
방문하는 state는 훈련 중 변하므로 이 프로파일은 사후에 복원되지 않는다.

실측된 모양이다(β=0.4, K=64이므로 지지집합 128칸).

| 위치 | 걸린 token 수 |
|---|---|
| pos0 | 64.0 |
| pos1 | 49.9 |
| pos2-3 | 47.2 |
| pos4-7 | 38.1 |
| pos8-15 | 32.8 |
| pos16+ | 24.5 |

앞쪽에 몰리는 모양이 사전 분석과 일치한다. pos0이 64.0인 것은 지지집합의 teacher 쪽 절반이
전부 걸렸다는 뜻으로, teacher가 첫 token에 확률을 몰아주고 anchor가 그것을 모를 때 나타나는
패턴이다. `target_floor_mass_max`가 0.39998로 관측된 것도 이론값 β에 도달한 사례다.

표는 하나다. `training/groups/success_counts`가 `{그룹 내 성공 수: 그룹 개수}` 히스토그램을 매
step 한 줄씩 쌓는다. DAPO의 `filtered_reward_counts` 패턴을 일반화한 로거를 썼고, verl의 `Counter`
병합 경로를 타서 iteration들이 합산된다. 스칼라 세 비율이 이 표에서 파생되며 그룹 내 성공률
분포도 여기서 정확히 나온다.

세 비율 중 informative는 나머지 둘의 여집합이지만 셋 다 남겼다. 계산 비용이 0이고, informative가
결론 지표이며, 셋을 나란히 봐야 informative의 증가가 all-fail 감소에서 온 것인지 구분된다.

## 8. 검증

단위 테스트는 문서의 정의와 성질을 불변식으로 삼는다(`Code/tests/`, 44개 통과).

정의 일치를 세 가지로 확인한다. toy 재현에서 Cost(0.4) = 0.0995, clamp된 token 2개, c = 0.9293이
나와 `toy_sims/floor_vs_kl.py`의 값과 소수점 넷째 자리까지 맞는다. clamp되지 않은 A:B odds가
teacher 6.0714에서 q* 6.0714로 보존된다. 확률 공간에서 독립적으로 푼 참조 구현과 최대 오차가
1.5×10⁻⁷다.

수학적 성질은 β ∈ {0.1, 0.2, 0.4, 0.8} 전부에서 확인한다. Σq* = 1(오차 ≤ 2.4×10⁻⁷), floor 제약
q*(v) ≥ β·π_A(v) 위반 0건, teacher 하한 q*(v) ≥ (1−β)·π_T(v) 위반 0건과 c ≥ 1−β(β=0.4에서 c 최소
0.6001), 같은 floor 보장에서 KL(q*‖π_T) ≤ KL(mixture‖π_T), 비용의 β 단조 증가, β→0에서 π_T로
수렴, 그리고 teacher가 한 token에 확률을 몰고 anchor가 그것을 무시할 때 q*가 그 token에 남기는
확률이 정확히 1−β다.

verl 통합은 커널이 packed 형식 `(1, total_nnz, V)`를 받아 gradient가 student logits까지 흐르는
것과, K = V일 때 전체 vocabulary 계산과 최대 오차 1.4×10⁻⁶으로 일치하는 것을 본다. 후자가 top-k
경로가 절단 없는 계산과 같음을 보이는 검사다.

실제 모델 검증은 teacher Qwen3-4B와 anchor Qwen3-1.7B-Base, anchor rollout에서 뽑은 state로 한다.

| β | Cost(β) | clamp된 token 수 | floor mass | c 최소 | floor 위반 |
|---|---|---|---|---|---|
| 0.1 | 0.0097 | 45,641 | 0.011 | 0.969 | 0 |
| 0.2 | 0.0326 | 55,276 | 0.037 | 0.926 | 0 |
| 0.4 | 0.1115 | 65,950 | 0.124 | 0.719 | 0 |
| 0.8 | 0.3921 | 79,222 | 0.314 | 0.256 | 0 |

Cost(0.4) = 0.1115는 Cal_Beta_Before_train.md의 실측 0.114와 같은 자릿수다. teacher가 14B가 아니고
state 표본이 작은데도 그렇다.

top-k 절단의 실제 영향도 같은 state에서 쟀다(β=0.4).

| K | teacher mass | anchor mass | 지켜진 floor |
|---|---|---|---|
| 64 | 0.9988 | 0.9837 | 98.5% |
| 128 | 0.9993 | 0.9901 | 99.1% |
| 512 | 0.9997 | 0.9969 | 99.7% |

K=128에서 floor 약속의 99.1%가 지켜진다. 남는 1% 미만은 anchor의 top-k 밖에 있는 token들이고,
그곳의 floor는 개별적으로 무의미한 크기다.

기존 경로의 회귀는 verl의 `tests/workers/test_distillation_topk_symmetry_on_cpu.py` 6개로 확인하고,
`ruff check`와 `ruff format`을 통과시킨다.

## 9. 알려진 한계

megatron 전략은 미지원이다. 그 커널은 vocab-parallel 샤드 위에서 손으로 쓴 backward를 쓰므로
target 교체에 별도 작업이 필요하고, 명시적 `NotImplementedError`로 막았다.

veomni의 fused kernel 경로도 미지원이다. `transformer_impl.py`의 고정 키 목록이 teacher 전용 세
필드만 꺼내므로 그 경로가 켜지면 anchor 진단이 빠진다. 우리가 쓰는 것은 그 아래의 일반
logit-processor 경로다.

top-k 절단은 남는다. anchor를 학습 프로세스에 두면 전체 vocabulary가 원리적으로 가능해지지만,
그러려면 anchor의 forward가 student의 손실 계산과 같은 micro-batch 안에서 일어나야 하고 그것은
FSDP 샤딩과 sequence parallel 분할의 가정을 다시 맞추는 일이 된다. K=512에서 이미 99.7%가
지켜지므로 남는 0.3%를 위해 학습 엔진 내부를 고칠 이유는 없다고 판단했다.
