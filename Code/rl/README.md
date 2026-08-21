# 훈련 런처

`Base -> distillation -> RL` 인계 실험을 돌린다. 골격은 verl 정본
`examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh` 이고, fsdp2 블록은 같은
디렉토리의 `run_qwen3_5_4b_fsdp.sh` 를 따랐다.

## 준비

```bash
conda activate ICLR-verl
PYTHONPATH=/home/eoeldroal python Code/rl/prepare_math.py      # 한 번만
```

MATH train 7,498문제에서 사전 분석이 쓴 200문제를 빼고 7,298개를 훈련에, test split에서
500개를 검증에 쓴다. 빼낸 200개는 `data/heldout.jsonl` 에 남아 branch panel 의 씨앗이 된다.
명령문 문자열은 `PreAnalysis/common.py:INSTRUCTION` 을 import 하므로 사전 분석과 갈라질 수 없다.

## 세 팔

```bash
ARM=floor   bash Code/rl/run_distill.sh      # relative-floor projection
ARM=vanilla bash Code/rl/run_distill.sh      # verl 내장 GKD OPD (대조군)
ARM=rl      bash Code/rl/run_distill.sh      # distillation 없이 Base 에서 GRPO
```

| | loss_mode | anchor | teacher pool | rollout.n |
|---|---|---|---|---|
| floor | `relative_floor_topk` | `anchor_from_ref` | TP 2 x 1 | 1 |
| vanilla | `forward_kl_topk` | 없음 (ref 도 안 만든다) | TP 2 x 1 | 1 |
| rl | 없음 | 없음 | 없음 | 16 |

`floor` 와 `vanilla` 는 GPU 4장(student 2 + teacher 2), `rl` 은 2장을 쓴다.

GPU 를 잡기 전에 인자 구성만 보려면 `DRY_RUN=1` 을 붙인다.

```bash
DRY_RUN=1 ARM=floor bash Code/rl/run_distill.sh
```

`rollout.n` 이 팔마다 다른 이유가 있다. 순수 distillation 에서 GRPO 그룹은 학습 신호가 아니고,
같은 prompt 를 여러 번 표집하는 것보다 더 많은 prompt 를 한 번씩 보는 쪽이 state 다양성이 크다.
정본 OPD 스크립트 둘 다 `n=1` 이다. RL 팔의 16 은 임의값이 아니라 `toy_sims/beta_design.py` 의
`G = 16` 이고, beta_knee 계산의 `N = 80 = G x T` 에 들어간 값이다.

## 프롬프트 형식

```bash
PROMPT_FORMAT=plain    # 기본. conf/prompt_format/plain.yaml
PROMPT_FORMAT=chat     # tokenizer 의 Qwen3 템플릿
```

Base anchor 를 쓰는 동안 `plain` 이어야 한다. 이유는 `conf/prompt_format/plain.yaml` 의 주석에
적었다. `chat` 으로 돌릴 때는 thinking 응답이 길어 `MAX_RESPONSE_LENGTH` 를 함께 키워야 한다.
`data.truncation` 은 프롬프트에만 적용되므로 응답은 터지지 않고 조용히 잘린다.

## 오프로딩

항상 켠다. `Document/Imp_Detail.md` 5절의 5단계 시분할이 verl 의 기본 동작이 아니라 이 인자들이
만드는 것이기 때문이다. `BaseEngineCtx._context_switch` 는 `param_offload` 와
`optimizer_offload` 가 둘 다 False 면 즉시 반환하고, 그러면 세 모델이 step 전체에 상주한다.

```
actor.fsdp_config.param_offload=True        # 생성 단계에 student 를 CPU 로
actor.fsdp_config.optimizer_offload=True    # 옵티마이저 상태는 학습 단계에만 GPU
actor.fsdp_config.offload_policy=False      # param_offload 와 배타적. peak 가 막힐 때만 True
ref.fsdp_config.param_offload=True          # 무효. forward_only 가 이미 강제한다
rollout.free_cache_engine=True              # student 가 내려간 뒤 KV 를 복구한다
```

`fsdp2` 를 쓰는 이유는 FSDP1 이 actor 의 native CPUOffload 를 강제로 끄기 때문이다
("causes incorrect results when using grad accumulation"). `use_dynamic_bsz=True` 가 곧 grad
accumulation 이므로 그 제약이 우리에게 그대로 걸린다. fsdp2 의 `CPUOffloadPolicy` 는
per-parameter sharding 이라 그 문제가 없고, 가중치 동기화에서도 모델 전체를 GPU 에 올리지 않는다
(`_skip_staging`).

`OFFLOAD_POLICY=True` 로 한 단계 더 갈 수 있다. 훈련 중에도 파라미터를 CPU 에 두므로 peak 가
가장 낮지만 매 layer 마다 fetch 비용을 물고, `param_offload` 의 수동 경로는 꺼진다. 큰 student 에서
peak 가 실제로 막힐 때만 켠다.

## 스케일을 올릴 때

| 인자 | 1.7B | 8B 이상 |
|---|---|---|
| `ROLLOUT_TP` | 1 | 2 이상 |
| `TRAIN_GPUS` | 2 | 늘린다 |
| `PPO_MAX_TOKEN_LEN_PER_GPU` | 24576 | 낮춘다 |
| `OFFLOAD_POLICY` | False | peak 가 막히면 True |
| `ROLLOUT_GPU_MEM_UTIL` | 0.4 | 낮춘다 |

`layered_summon` 은 쓰지 않는다. 추적해 보면 LoRA 경로에서만 쓰이고
(`transformer_impl.py:981` -> `collect_lora_params`) 우리는 비-LoRA + sglang 이라 아무 일도
하지 않는다.

## 환경

`VERL_USE_UV=0` 이 스크립트에 박혀 있다. 정본은 `uv run --frozen` 으로 committed `uv.lock`
(torch 2.11 + cu130) 을 쓰는데 이 서버 드라이버가 CUDA 12.8 까지라 그 락이 맞지 않는다.
확정된 환경은 `Code/verl_smoke/README.md` 에 있다.
