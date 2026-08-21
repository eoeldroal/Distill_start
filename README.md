# Distillation → RL 인계의 탐색 구조와 회복 가능성

On-policy distillation이 downstream RL에 넘기는 탐색 상태를 측정하고, 그 손상이
어디까지 복구되는지를 보는 연구다. Distillation 직후의 benchmark score가 아니라
다음 stage가 물려받는 branch 구조를 관측 대상으로 삼는다.

측정은 세 관측량으로 한다. token 수준 불확실성 `H`, branch 진입 분포 `E`,
그리고 branch를 택한 rollout이 정답에 도달하는 비율 `V`다. 여기서 두 요약지표
Effective Branch Breadth와 Budgeted Successful Branch Discovery를 만들어
Base → Distillation → RL을 같은 branch space에서 추적한다.

Vanilla distillation이 student가 원래 갈 수 있던 branch의 확률을 지나치게 깎는
경우에 대비해 **relative-floor projection**을 함께 구현했다.

```
q* = argmin_q KL(q ‖ π_T)   s.t.   q(v) ≥ β · π_A(v)  for all v
   ⇒ q*(v) = max(c · π_T(v), β · π_A(v))
```

Teacher를 최대한 따라가되, frozen anchor `π_A`가 주던 확률의 β배 아래로는 어떤
candidate도 내려보내지 않는다. Floor가 절대 확률이 아니라 anchor 확률에 대한
비율이므로 보호의 크기는 token마다 anchor의 판단을 따라간다.

## 디렉토리

| 경로 | 내용 |
|---|---|
| `Document/` | 논문 초안(`Draft.md`), 실험 설계(`Experiments.md`), β 설계 분석(`Cal_Beta_Before_train.md`), 구현 부록(`Imp_Detail.md`) |
| `Code/verl/` | verl을 vendor한 것. 우리 수정이 여기 들어 있다 |
| `Code/RELATIVE_FLOOR.md` | 구현 기록. 무엇을 바꿨고 무엇으로 검증했는지 |
| `Code/tests/` | relative-floor 커널과 anchor 경로 검증 (44개) |
| `Code/verl_smoke/` | 환경 구축 기록, 스모크 스크립트, 실행 로그 |
| `Experiment/PreAnalysis/` | 훈련 전 Cost(β) 측정. frozen 모델 두 개의 forward pass만 쓴다 |
| `Experiment/BranchDev/` | branch 탐색 도구와 pilot 생성 |

## Code/verl 에 대해

Upstream `verl-project/verl` 을 `b256ebf8` (v0.8.0-328) 에서 통째로 가져왔다.
이력을 두 커밋으로 쪼개 두었으므로 우리 기여만 따로 뽑아낼 수 있다.

```bash
# 우리 수정만 보기 (15개 파일 +807 -59)
git diff $(git log --format=%h --grep="^vendor: verl" -1) HEAD -- Code/verl
```

바꾼 것은 14개 파일 +725줄과 신규 `verl/trainer/distillation/projection.py` 82줄이다.
`loss_mode="relative_floor_topk"` 로 선택하며, 기존 `forward_kl_topk` 경로는
건드리지 않았다.

두 가지를 알아 둘 것이 있다. upstream verl은 `recipe/` 를 별도 submodule
(`verl-project/verl-recipe`)로 두는데 우리는 초기화하지 않았으므로 그 디렉토리가
비어 있다. 그리고 vendor를 택했으므로 upstream 이력은 이 저장소에 없다.
새 upstream으로 옮길 때는 그 커밋 위에 위 diff를 다시 적용하는 방식이 된다.

## 환경

conda env `ICLR-verl` 을 쓴다. 이 서버 드라이버가 CUDA 12.8까지만 지원해서
verl 공식 락(torch 2.11 + cu130)을 쓸 수 없고, torch 2.9.1+cu128 · flash-attn 2.8.3 ·
sglang 0.5.9 조합으로 맞췄다. 구축 과정과 확정 버전은 `Code/verl_smoke/README.md`,
그때의 pip freeze는 `Code/verl_smoke/env_snapshot.txt` 에 있다.

`Experiment/BranchDev/` 의 생성 스크립트는 OpenRouter를 쓴다. `.env.example` 을
`.env` 로 복사해 키를 채운다. `.env` 는 추적하지 않는다.

## 추적하지 않는 것

재계산이 싼 대용량 산출물은 이력에 남기지 않는다. 로컬 임베딩 모델로 다시 만들 수
있는 `Experiment/BranchDev/outputs/emb_*.npy`, 전처리 스크립트가 다시 만드는
`Code/verl_smoke/data/`, ray와 hydra가 실행마다 새로 쓰는
`Code/verl_smoke/outputs/` 가 그렇다.

반대로 OpenRouter 생성물은 다시 만들면 돈이 드므로 그대로 커밋한다.
스모크 로그도 남긴다. `Document/Imp_Detail.md` 와 `Code/RELATIVE_FLOOR.md` 가
인용하는 측정치의 출처가 그 로그들이다.
