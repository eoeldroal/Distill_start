# verl 환경 구축과 스모크 테스트 (2026-08-20)

conda env **`ICLR-verl`** 에서 verl 0.10.0.dev를 동작시킨 기록. 이 서버의 드라이버가
CUDA 12.8까지만 지원하므로 verl 공식 락(torch 2.11 + cu130)을 쓸 수 없어, 대신
torch 2.9.1 + cu128 조합으로 손수 맞췄다.

## 확정된 환경

| 항목 | 버전 | 비고 |
|---|---|---|
| python | 3.12.13 | conda env `ICLR-verl` |
| torch | 2.9.1+cu128 | 드라이버 570.124.06 / CUDA 12.8과 맞음 |
| flash-attn | 2.8.3 | `cu12torch2.9cxx11abiTRUE-cp312` wheel, `libcudart.so.12` 확인 |
| sglang | 0.5.9 | verl이 0.5.5까지 하위 호환 분기를 갖고 있어 동작 |
| sgl-kernel | 0.3.21 | verl의 `assert_pkg_version("sgl_kernel","0.1.1")` 통과 |
| flashinfer-python | 0.6.3 | sglang attention backend |
| transformers | 4.57.1 | |
| ray | 2.57.0 | |
| tensordict | 0.10.0 | verl 허용 범위 `>=0.8,<=0.10` 상한 |
| TransferQueue | 0.1.10.dev0 | v1 trainer 필수 (`main_ppo`가 import) |
| verl | 0.10.0.dev0 | `pip install -e . --no-deps` (의존성 해석 우회) |

전체 목록은 `env_snapshot.txt`.

### 왜 `--no-deps` 인가
verl의 `pyproject.toml`은 모든 GPU 백엔드를 torch 2.11 + cu130으로 고정하고 torch를
cu130 인덱스로 라우팅한다. 그대로 설치하면 CUDA 13용 torch가 우리 것을 덮어써
GPU가 잡히지 않는다. 그래서 쓸 수 있는 조합을 먼저 깔고 verl은 코드만 얹었다.

### 쓸 수 없는 것
- verl wheelhouse의 flash-attn: `libcudart.so.13`(CUDA 13) 링크 → 이 드라이버에서 불가
- megatron / torchtitan / veomni / nemo_automodel: 미설치 (우리는 FSDP만 쓰므로 무관)

## 스모크 테스트 결과

### 1. sglang 단독 (`test_sglang.py`)
Qwen3-1.7B-Base 기동, canonical sampler(`top_k=-1, top_p=1.0, temperature=1.0`)로 생성 정상.
verl의 teacher 채점이 쓰는 `prompt_logprobs`(return_logprob + top_logprobs_num)도 정상 반환.

### 2. GRPO 학습 (`run_grpo_smoke.sh`, 2 step)
GSM8K, Qwen3-1.7B-Base, sglang rollout + FSDP, G=16, GPU 2장. 로그 `grpo_smoke.log`.

| 지표 | 값 | 의미 |
|---|---|---|
| `rollout_actor_probs_pearson_corr` | 0.9995 | rollout 분포와 학습 분포 일치 = canonical sampler 의도대로 작동 |
| `rollout_corr/kl` | 0.0014 | 두 분포 사이 KL이 사실상 0 |
| `actor/grad_norm` | 0.372 | 학습 신호 정상 |
| `perf/mfu/actor` | 0.23 | flash-attn 사용 중 |
| step 시간 | 9~45초 | |

### 3. OPD = on-policy distillation (`run_opd_smoke.sh`, 2 step)
**우리 방법의 삽입 지점이 되는 경로.** student Qwen3-1.7B-Base(GPU 2장) +
teacher Qwen3-4B(GPU 1장, sglang 서버), `loss_mode=forward_kl_topk`, `topk=64`,
`use_policy_gradient=False`(GKD 형태). 로그 `opd_smoke.log`.

| 지표 | 값 |
|---|---|
| `distillation/loss` | 0.557 → 0.718 |
| `distillation/teacher_mass` | 0.966 (top-64가 담은 teacher 확률) |
| `distillation/student_mass` | 0.877 |
| `distillation/overlap_ratio` | 0.588 |
| `actor/grad_norm` | 8.12 |

teacher가 별도 GPU 풀에서 student rollout을 채점하고, 그 top-k 분포로 손실이
계산되어 gradient가 흐르는 것까지 확인됐다.

### 4. q* 삽입 지점 검증 (`test_qstar_insertion.py`)
verl의 KL kernel `verl.trainer.distillation.fsdp.losses.kl_divergence`에 target으로
q*를 넣는 형태를 검증.

- toy 정합성: Cost(0.4) = 0.0995 (`Document/toy_sims/floor_vs_kl.py` 기준값과 일치),
  clamp되지 않은 token 사이 A:B odds = 6.071로 teacher와 동일 (odds 보존 성질)
- teacher/anchor top-512 합집합 위에서 q* 생성 → verl kernel로 손실 → student logits까지
  gradient 흐름 확인
- floor 보장: 현실적인 뾰족함의 분포에서 anchor top-512 mass 0.9997, 지켜진 floor 100%

### 5. 실제 모델 end-to-end (`test_real_models_qstar.py`)
teacher Qwen3-4B, anchor Qwen3-1.7B-Base, plain 프롬프트 state 2개.

| β | Cost(β) | clamp된 token 수 | floor mass |
|---|---|---|---|
| 0.1 | 0.030 | 46,460 | 0.037 |
| 0.2 | 0.101 | 59,797 | 0.085 |
| 0.4 | 0.316 | 75,626 | 0.248 |
| 0.8 | 0.990 | 96,461 | 0.520 |

성질 전부 성립: Σq* = 1, c ≥ 1−β, floor 위반 0건.
(Cost 값은 사전 분석의 teacher(Qwen3-14B)와 다른 4B teacher, state 2개짜리
표본이므로 `Cal_Beta_Before_train.md`의 0.114와 직접 비교할 수치는 아니다.)

## relative-floor 실행 (2026-08-21 추가)

우리 방법을 실제로 돌리는 두 스크립트. 자세한 내용은 `../RELATIVE_FLOOR.md`.

| 스크립트 | anchor 를 어디에 두나 | GPU |
|---|---|---|
| `run_floor_ref_smoke.sh` | reference policy (actor 와 같은 GPU) — **권장** | 3장 |
| `run_floor_smoke.sh` | 전용 sglang 서버 | 4장 |

둘은 동일한 target 과 손실을 낸다(`../tests/test_anchor_equivalence_e2e.py`, 오차 0).

로그 파일: `floor_ref.log`(anchor_from_ref), `floor_smoke.log`(anchor_model),
`cmp_*.log`/`g_*.log`(두 방식 비교 실행), `opd_smoke.log`(기존 OPD), `grpo_smoke.log`(GRPO).

주의: `pkill -f "main_ppo"` 는 **자기 셸을 죽인다**(명령줄에 그 문자열이 있어 자기 자신에
매칭된다). `pkill -9 -f "[m]ain_ppo"` 처럼 대괄호 트릭을 쓴다.

## 실행 방법

```bash
conda activate ICLR-verl
cd Experiment/verl_smoke

CUDA_VISIBLE_DEVICES=3 python test_sglang.py              # sglang 단독
CUDA_VISIBLE_DEVICES=3,4 ./run_grpo_smoke.sh              # GRPO 2 step
CUDA_VISIBLE_DEVICES=3,4,5 ./run_opd_smoke.sh             # OPD 2 step
CUDA_VISIBLE_DEVICES=3 python test_qstar_insertion.py     # q* 삽입 지점
CUDA_VISIBLE_DEVICES=3 python test_real_models_qstar.py   # 실제 모델 q*
```

주의: `VERL_USE_UV=0`을 넘겨야 uv 대신 현재 conda env를 쓴다 (스크립트에 포함).
sglang은 multiprocessing spawn을 쓰므로 파이썬 테스트에 `if __name__ == "__main__":`
가드가 필수다.

## 남은 확인 사항

- torch 2.11을 요구하는 verl 코드 경로를 아직 만나지 않았다. 스모크 범위(FSDP+sglang+OPD)
  에서는 2.9.1로 충분하다. 더 깊은 기능에서 문제가 나면 flash-attn 소스 빌드로 2.11 전환.
- teacher를 Qwen3-14B로 올릴 때의 GPU 배치와 메모리는 아직 미측정.
- 우리 손실(relative-floor projection) 등록과 anchor 채점 경로 추가는 다음 작업.
