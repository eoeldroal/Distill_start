#!/usr/bin/env bash
# GRPO 스모크 테스트: Qwen3-1.7B-Base + sglang rollout + FSDP, GSM8K 2 step.
# 우리 연구 설정을 반영: canonical sampler(top_k=-1, top_p=1.0), G=16.
set -xeuo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
export HF_HUB_OFFLINE=1
export VERL_USE_UV=0                       # uv 대신 현재 conda env 사용
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3,4}
export WANDB_MODE=offline

NGPUS=$(python3 -c "import os;print(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))")

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${HERE}/data/train.parquet" \
  data.val_files="${HERE}/data/test.parquet" \
  data.train_batch_size=8 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=Qwen/Qwen3-1.7B-Base \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  trainer.logger='["console"]' \
  trainer.project_name=verl_smoke \
  trainer.experiment_name=grpo_qwen3_1p7b_base \
  trainer.n_gpus_per_node=${NGPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_training_steps=2 \
  trainer.default_local_dir="${HERE}/ckpt" \
  "$@"
