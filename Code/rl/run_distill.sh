#!/usr/bin/env bash
# Base -> distillation -> RL 인계 실험의 런처. 세 팔을 한 스크립트로 돌린다.
#
#   ARM=floor    relative-floor projection.  q* = max(c*pi_T, beta*pi_A)
#   ARM=vanilla  verl 내장 GKD OPD.          target = teacher
#   ARM=rl       distillation 없이 GRPO.     Base 에서 바로 RL
#
# 골격은 examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh 를 따르고,
# fsdp2 관련 블록은 같은 디렉토리의 run_qwen3_5_4b_fsdp.sh 를 따른다. 정본과 갈리는 인자는
# 아래에 이유를 적었다.
set -xeuo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

# verl 정본은 VERL_USE_UV=1 이 기본이라 committed uv.lock (torch 2.11 + cu130) 으로 돌리는데,
# 이 서버 드라이버는 CUDA 12.8 까지라 그 락이 맞지 않는다. conda env ICLR-verl 을 쓴다.
export VERL_USE_UV=0
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

# ---- user-adjustable ----
ARM=${ARM:-floor}                                   # floor | vanilla | rl
PROMPT_FORMAT=${PROMPT_FORMAT:-plain}               # plain | chat

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-1.7B-Base}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-14B}

# floor 의 강도. beta_knee = ln(1/(1-c))/(m*N) = ln20/(0.10*80) ~ 0.37 의 정본화 값.
BETA=${BETA:-0.4}

# 정본은 64. floor 준수율이 anchor top-k 커버리지에 묶이고(K=64 에서 98.5%, 128 에서 99.1%)
# teacher mass 는 64 에서 이미 0.9988 로 포화하므로, K 를 키워 얻는 것은 전부 floor 쪽이다.
TOPK=${TOPK:-128}

NNODES=${NNODES:-1}
TRAIN_GPUS=${TRAIN_GPUS:-2}                         # student pool (actor + ref + rollout colocate)
TEACHER_TP=${TEACHER_TP:-2}                         # 14B bf16 28GB -> 카드당 14GB
# 단일 teacher 의 num_replicas 는 verl 이 pool_size // per_replica 로 직접 계산한다
# (_resolve_teacher_models). 그래서 이 값은 pool 크기를 정하는 입력일 뿐이고 인자로 넘기지 않는다.
TEACHER_REPLICAS=${TEACHER_REPLICAS:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}     # train 과 같게 두면 배치당 업데이트 1회
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
ACTOR_LR=${ACTOR_LR:-1e-6}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.4}

# distillation 팔은 정본 두 스크립트와 같이 1. 순수 GKD 에서 그룹은 학습 신호가 아니고,
# 같은 prompt 를 여러 번 표집하는 것보다 더 많은 prompt 를 한 번씩 보는 쪽이 state 다양성이
# 크다. RL 팔은 beta 유도가 쓴 G=16 을 맞춘다 (beta_design.py 의 G).
ROLLOUT_N_DISTILL=${ROLLOUT_N_DISTILL:-1}
ROLLOUT_N_RL=${ROLLOUT_N_RL:-16}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}      # 비우면 epochs 로 돈다
SAVE_FREQ=${SAVE_FREQ:-200}                         # E/V/B/D_succ 가 얼린 checkpoint 에서 계산된다
TEST_FREQ=${TEST_FREQ:-5}

# offload_policy 는 param_offload 와 배타적이다 (verl 이 켜지면 수동 경로를 끈다). 훈련 중에도
# 파라미터를 CPU 에 두므로 매 layer fetch 비용을 물고, 정본 fsdp2 스크립트도 False 다.
# peak 가 실제로 막힐 때만 켠다.
OFFLOAD_POLICY=${OFFLOAD_POLICY:-False}

LOGGER=${LOGGER:-'["console","wandb"]'}
PROJECT_NAME=${PROJECT_NAME:-distill_start}
# ---- end user-adjustable ----

case "$ARM" in floor|vanilla|rl) ;; *) echo "ARM must be floor|vanilla|rl" >&2; exit 1 ;; esac
case "$PROMPT_FORMAT" in plain|chat) ;; *) echo "PROMPT_FORMAT must be plain|chat" >&2; exit 1 ;; esac

TRAIN_FILE=${TRAIN_FILE:-${HERE}/data/train.parquet}
VAL_FILE=${VAL_FILE:-${HERE}/data/test.parquet}
for f in "$TRAIN_FILE" "$VAL_FILE"; do
    [ -f "$f" ] || { echo "missing $f; run prepare_math.py first" >&2; exit 1; }
done

MAX_NUM_TOKENS=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))
TEACHER_POOL=$(( TEACHER_TP * TEACHER_REPLICAS ))
case "$ARM" in
    floor)   AUTO_NAME="floor_${PROMPT_FORMAT}_b${BETA}_k${TOPK}" ;;
    vanilla) AUTO_NAME="vanilla_${PROMPT_FORMAT}_k${TOPK}" ;;
    rl)      AUTO_NAME="rl_${PROMPT_FORMAT}_n${ROLLOUT_N_RL}" ;;
esac
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$AUTO_NAME}

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_FILE}']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=False                              # 팔 간 matched protocol. 정본 OPD 도 False
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.use_remove_padding=True  # 우리 커널이 packed (1,total_nnz,V) 를 받는다
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

# fsdp2 를 쓰는 이유: FSDP1 은 actor 의 native CPUOffload 를 강제로 끈다
# ("causes incorrect results when using grad accumulation"). use_dynamic_bsz=True 가 곧
# grad accumulation 이므로 그 제약이 우리에게 그대로 걸린다. fsdp2 의 CPUOffloadPolicy 는
# per-parameter sharding 이라 그 문제가 없고, 가중치 동기화도 모델 전체를 GPU 에 올리지 않는다.
ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False        # OPD 에선 ref KL 을 끈다 (opd.md)
    actor_rollout_ref.actor.entropy_coeff=0          # entropy 보너스는 측정 대상에 직접 개입한다
    actor_rollout_ref.actor.use_torch_compile=True   # entropy_from_logits 만 compile 한다
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.offload_policy=${OFFLOAD_POLICY}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
)

# ref = anchor. forward_only:True 가 offload 를 강제하므로 param_offload 는 사실 무효지만,
# anchor 가 CPU 상주라는 것을 스크립트에 남긴다.
#
# strategy 는 넘기지 않는다. ref/ref.yaml 이 이미
# `strategy: ${actor_rollout_ref.actor.strategy}` 로 actor 를 따라가므로, 명시하면 두 번째
# 진실 원천이 생겨 나중에 actor 만 바꿀 때 조용히 갈린다. 정본 fsdp2 스크립트는 둘 다 적지만
# 그건 중복이다.
REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.offload_policy=False
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang            # 정본은 vllm. 우리 env 가 sglang 0.5.9
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS}
    actor_rollout_ref.rollout.temperature=1.0        # on-policy 분포를 왜곡하지 않는다
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.free_cache_engine=True # student 가 CPU 로 내려간 뒤 KV 를 복구한다
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.calculate_log_probs=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger="${LOGGER}"
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${TRAIN_GPUS}
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir="${HERE}/ckpt/${EXPERIMENT_NAME}"
)
[ -n "$TOTAL_TRAINING_STEPS" ] && TRAINER+=( trainer.total_training_steps=${TOTAL_TRAINING_STEPS} )

TEACHER=(
    distillation.enabled=True
    distillation.nnodes=${NNODES}
    distillation.n_gpus_per_node=${TEACHER_POOL}     # num_replicas * world_size 와 정확히 같아야 한다
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}"
    distillation.teacher_models.teacher_model.inference.name=sglang
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_NUM_TOKENS}
    distillation.distillation_loss.topk=${TOPK}
    distillation.distillation_loss.use_task_rewards=False   # distillation 과 RL 을 분리한다
    distillation.distillation_loss.use_policy_gradient=False # top-k 분포 신호를 PG 로 낭비하지 않는다
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

case "$ARM" in
    floor)
        ARM_ARGS=(
            "${TEACHER[@]}"
            "${REF[@]}"
            distillation.anchor_from_ref=True
            distillation.distillation_loss.loss_mode=relative_floor_topk
            distillation.distillation_loss.relative_floor_beta=${BETA}
            actor_rollout_ref.rollout.n=${ROLLOUT_N_DISTILL}
        )
        ;;
    vanilla)
        # ref 를 만들지 않는다. anchor 가 없으므로 정본 GKD OPD 그대로다.
        ARM_ARGS=(
            "${TEACHER[@]}"
            distillation.distillation_loss.loss_mode=forward_kl_topk
            actor_rollout_ref.rollout.n=${ROLLOUT_N_DISTILL}
        )
        ;;
    rl)
        # teacher pool 도 anchor 도 없다. Base 에서 바로 GRPO.
        ARM_ARGS=(
            distillation.enabled=False
            actor_rollout_ref.rollout.n=${ROLLOUT_N_RL}
        )
        ;;
esac

########################### launch ###########################
ARGV=(
    --config-dir "${HERE}/conf"
    "+prompt_format=${PROMPT_FORMAT}"
    "${DATA[@]}"
    "${MODEL[@]}"
    "${ACTOR[@]}"
    "${ROLLOUT[@]}"
    "${TRAINER[@]}"
    "${ARM_ARGS[@]}"
    ray_kwargs.ray_init.runtime_env.py_executable=null
    "$@"
)

# DRY_RUN=1 이면 인자 구성만 찍고 끝낸다. GPU 를 잡기 전에 팔마다 무엇이 넘어가는지 확인한다.
if [ "${DRY_RUN:-0}" != 0 ]; then
    set +x
    echo "ARM=${ARM}  PROMPT_FORMAT=${PROMPT_FORMAT}  GPUs=${TRAIN_GPUS} train"
    [ "$ARM" != rl ] && echo "                                    + ${TEACHER_POOL} teacher"
    printf '  %s\n' "${ARGV[@]}"
    exit 0
fi

mkdir -p "${HERE}/logs"
LOG="${HERE}/logs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
python3 -m verl.trainer.main_ppo "${ARGV[@]}" 2>&1 | tee "${LOG}"
