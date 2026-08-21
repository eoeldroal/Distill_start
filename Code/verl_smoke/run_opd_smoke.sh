#!/usr/bin/env bash
# OPD(on-policy distillation) 스모크 테스트 — 우리 방법의 삽입 지점이 되는 경로.
# student: Qwen3-1.7B-Base (GPU 2장), teacher: Qwen3-4B (GPU 1장, sglang 서버)
# 우리 손실은 아직 없고, verl 기존 forward_kl_topk(GKD 형태)로 파이프라인 성립만 확인한다.
set -xeuo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export HF_HUB_OFFLINE=1
export VERL_USE_UV=0
export WANDB_MODE=offline
STUDENT=${STUDENT:-Qwen/Qwen3-1.7B-Base}
TEACHER=${TEACHER:-Qwen/Qwen3-4B}
STUDENT_GPUS=${STUDENT_GPUS:-2}
TEACHER_GPUS=${TEACHER_GPUS:-1}
TOPK=${TOPK:-64}
LOSS_MODE=${LOSS_MODE:-forward_kl_topk}
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${HERE}/data/train.parquet" \
  data.val_files="${HERE}/data/test.parquet" \
  data.train_batch_size=4 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=${STUDENT} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  distillation.enabled=True \
  distillation.n_gpus_per_node=${TEACHER_GPUS} \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path=${TEACHER} \
  distillation.teacher_models.teacher_model.inference.name=sglang \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.55 \
  distillation.distillation_loss.loss_mode=${LOSS_MODE} \
  distillation.distillation_loss.topk=${TOPK} \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=False \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  trainer.logger='["console"]' \
  trainer.project_name=verl_smoke \
  trainer.experiment_name=opd_qwen3_1p7b_base \
  trainer.n_gpus_per_node=${STUDENT_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_training_steps=2 \
  trainer.default_local_dir="${HERE}/ckpt_opd" \
  "$@"
