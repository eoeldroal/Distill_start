# 훈련

```bash
conda activate ICLR-verl
cd /home/eoeldroal/WorkPlace/Distill_start

bash Code/rl/run_opd.sh      # baseline: verl 내장 GKD OPD
bash Code/rl/run_ember.sh    # EMBER: relative-floor projection
```

두 스크립트가 각각 인자 전체를 담고 있다. `diff run_opd.sh run_ember.sh` 가 방법의 차이 전부다.

| | run_opd.sh | run_ember.sh |
|---|---|---|
`loss_mode` | `forward_kl_topk` | `relative_floor_topk` |
`relative_floor_beta` | | 0.4 |
`anchor_from_ref` | | True |
그 외 78줄 | 동일 | 동일 |

GPU 8장. student pool 4장(actor + ref + rollout colocate), teacher pool 4장(Qwen3-14B TP1 × 4 replica).
13,976문제 배치 128이면 1 epoch = 109 스텝이고 `save_freq=20` 이라 checkpoint 5개가 남는다.

한 번만 값을 바꿔 돌리려면 뒤에 Hydra override 를 붙인다.

```bash
bash Code/rl/run_ember.sh distillation.distillation_loss.relative_floor_beta=0.2
```

## 데이터

```bash
PYTHONPATH=/home/eoeldroal python Code/rl/prepare_cascade.py
```

`nvidia/Nemotron-Cascade-RL-Math` 14,476문제를 train 13,976 / val 500 으로 나눠
`data/{train,test}.parquet` 에 쓴다. 필드는 넷이다.

```
data_source  = "nemotron_cascade_math"      # 채점 함수를 고르는 키. prime_math 로 간다
prompt       = [{"role": "user", ...}]      # 문제 + PreAnalysis/common.py 의 INSTRUCTION
reward_model = {"style": "rule", "ground_truth": answer}
extra_info   = {"split", "index", "source"}
```

`ability` 는 넣지 않는다. verl 이 읽지 않는다.

`data_source` 문자열은 `verl/utils/reward_score/__init__.py` 의 numina 목록에 추가해 뒀다.
목록에 없으면 첫 채점에서 `NotImplementedError` 가 난다.

## 데이터셋을 이걸로 정한 근거

```bash
PYTHONPATH=/home/eoeldroal python Code/rl/probe_difficulty.py --phase generate
PYTHONPATH=/home/eoeldroal python Code/rl/probe_difficulty.py --phase score
```

300문제 × 4샘플, 평문 프롬프트, 예산 8192. 생성과 채점을 나눈 이유는 sglang 이 설치한
multiprocessing spawn 안에서 `prime_math` 의 sympy 동등성 검사가 조용히 전부 실패하기
때문이다. 생성물을 전문 저장하므로 채점은 몇 번이든 다시 할 수 있다.

| | pass@1 | pass@4 | all_fail | 응답 길이 | 절단율 |
|---|---|---|---|---|---|
1.7B-Base 평문 | 0.069 | 0.177 | 0.823 | 1,092 | 0.008 |
1.7B-Post 평문 | 0.128 | 0.323 | 0.677 | 8,108 | 0.983 |
14B-Post 평문 | **0.312** | 0.473 | 0.527 | 5,769 | 0.490 |
14B-Post chat | 0.305 | 0.580 | 0.420 | 5,830 | 0.491 |

teacher 가 31% 풀고 student 가 7% 푼다. 증류가 옮길 여지가 4.5배이고, 이게 데이터셋을
고른 기준이다. MATH 는 이 student 에게 너무 쉬웠다.

채점자는 `prime_math` 여야 한다. 같은 생성물을 `math_dapo` 로 채점하면 14B 가 0.112 로
떨어진다. `14` 와 `14.0` 을 다르게 보는데 Cascade 의 답에 `999.998976` 같은 값이 있다.

평문과 chat 의 teacher pass@1 이 0.312 대 0.305 로 갈리지 않는다. 두 형식 모두 절단율이
0.49 라 14B 의 병목이 형식이 아니라 추론 길이이기 때문이다. 평문을 쓰는 이유는 teacher 가
아니라 anchor 다.

## 프롬프트 형식

`conf/prompt_format/plain.yaml` 이 chat template 을 평문 통과 템플릿으로 덮어쓴다.
Qwen3 템플릿은 `add_generation_prompt` 에서 `<|im_start|>assistant\n` 으로 끝나고 사고
블록을 열어 두므로, teacher 는 그 자리에서 `<think>`(151667)를 확률 ~1 로 내지만 Base
anchor 는 그 token 을 본 적이 없어 H 3.94 로 퍼진다. floor 가 그 질량의 β배를 지키려 하면
Cost(β) 가 0.277 에서 9.97 nats 가 되고, 실측에서 pos0 의 floor binding 이 128/128 로
포화했다.

## 오프로딩

`param_offload` 와 `optimizer_offload` 를 항상 켠다. `Document/Imp_Detail.md` 5절의 5단계
시분할이 verl 의 기본 동작이 아니라 이 두 인자가 만드는 것이다. `BaseEngineCtx._context_switch`
는 둘 다 False 면 즉시 반환하고, 그러면 student 와 rollout 이 step 전체에 상주한다.
anchor 쪽은 `forward_only: true` 가 강제하므로 인자가 필요없다.

`fsdp2` 를 쓰는 이유는 FSDP1 이 actor 의 native CPUOffload 를 강제로 끄기 때문이다
("causes incorrect results when using grad accumulation"). `use_dynamic_bsz=True` 가 곧
grad accumulation 이라 그 제약이 그대로 걸린다.

## 환경

`VERL_USE_UV=0` 이 두 스크립트에 박혀 있다. verl 정본은 uv 로 committed `uv.lock`
(torch 2.11 + cu130) 을 쓰는데 이 서버 드라이버가 CUDA 12.8 까지다. 확정된 환경은
`Code/verl_smoke/README.md` 에 있다.
