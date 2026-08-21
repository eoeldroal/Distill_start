#!/usr/bin/env bash
# Baseline. verl 내장 on-policy distillation, GKD 변형 (docs/algo/opd.md).
set -euo pipefail
export VERL_USE_UV=0
export HF_HUB_OFFLINE=1

HERE=$(dirname "$0")
NAME=opd
mkdir -p "${HERE}/logs"

python3 -m verl.trainer.main_ppo \
    --config-dir "${HERE}/conf" \
    +prompt_format=plain \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="['${HERE}/data/train.parquet']" \
    data.val_files="['${HERE}/data/test.parquet']" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    data.seed=42 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B-Base \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.actor.fsdp_config.seed=42 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=3073 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    distillation.enabled=True \
    distillation.nnodes=1 \
    distillation.n_gpus_per_node=4 \
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-14B \
    distillation.teacher_models.teacher_model.inference.name=sglang \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.85 \
    distillation.teacher_models.teacher_model.inference.max_model_len=3073 \
    distillation.distillation_loss.loss_mode=forward_kl_topk \
    distillation.distillation_loss.topk=128 \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=False \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    trainer.balance_batch=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=distill_start \
    trainer.experiment_name=${NAME} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${HERE}/ckpt/${NAME}" \
    ray_kwargs.ray_init.runtime_env.py_executable=null \
    "$@" 2>&1 | tee "${HERE}/logs/${NAME}_$(date +%Y%m%d_%H%M%S).log"
