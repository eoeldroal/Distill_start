# BranchDev 실행 안내

이 디렉터리에는 branch panel을 만드는 생성·분석 도구와 원자료가 있다. Branch와 \(E\)의
정의, 전체 실험 순서는 [Branch Panel and E](../../Document/Branch_Panel_and_E.md)를 따른다. 이 문서는 API 설정,
스크립트, 명령어, 파일 형식만 다룬다.

## 환경

로컬 모델 스크립트는 conda 환경 `ICLR-verl`을 사용한다. OpenRouter 스크립트는 저장소 루트의
`.env`에서 키를 읽는다.

```text
OPENROUTER_API_KEY="sk-or-v1-..."
```

`.env`는 Git으로 추적하지 않는다. 예시는 루트 `.env.example`에 있다.

## API 모델과 endpoint

| 모델 | endpoint |
|---|---|
| `z-ai/glm-5.2` | `ambient/fp8` |
| `deepseek/deepseek-v4-flash-0731` | `deepinfra/fp8` |
| `qwen/qwen3.8-27b` | `chutes/fp8` |
| `minimax/minimax-m3` | `deepinfra/fp8` |
| `xiaomi/mimo-v2.5-pro` | `deepinfra/fp8` |
| `meta/muse-glimmer-30b` | `deepinfra/bf16` |

모든 요청에서 endpoint를 고정하고 `allow_fallbacks: false`를 사용한다. OpenRouter endpoint가
`top_p`나 `top_k`를 지원 목록에 선언하지 않으면 값을 보내도 조용히 버릴 수 있다. 응답의
`provider` 필드를 저장해 실제 endpoint를 확인한다.

Rate limit은 같은 모델의 endpoint끼리 공유할 수 있다. 동시성을 낮추거나 endpoint만 바꿔서는
모델 단위 제한이 풀리지 않는다.

## 공통 prompt

```text
<문제>

Please reason step by step, and put your final answer within \boxed{}.
```

기존 API pilot은 temperature 1.3, top-p 1.0, top-k 0, max_tokens 256으로 reasoning 앞부분만
모았다. `outputs/discovery_pilot_v2.jsonl`에는 요청 1,920건이 있고, 이 중 1,913건이 성공한
trajectory다. 최종 panel용 API 자료는 같은 여섯 모델을 문제당 16회 호출하되, 각 모델이 답까지
완결할 수 있는 token budget을 사용해야 한다. 현재 `run_discovery.py`는 256-token pilot
설정이므로 최종 API 생성 전에 완결 풀이용 인자를 추가해야 한다.

## Base recursive tree

현재 생성기는 `branch_tree.py`다. Opening의 첫 두 위치에서는 1% 이상 token을 모두 열고,
그 뒤에는 entropy와 candidate 수를 함께 만족하는 internal fork를 depth까지 재귀 확장한다.

```bash
/home/eoeldroal/miniconda3/bin/conda run -n ICLR-verl \
  python Experiment/BranchDev/branch_tree.py \
  --problems 180 \
  --thresh 0.01 \
  --entropy-thresh 1.5 \
  --depth 3 \
  --follow 48 \
  --device cuda:3 \
  --tag base_d3
```

여러 문제를 지정할 때는 쉼표로 연결한다. `--problems`를 생략하면 현재 panel 후보 8개를 모두
사용한다.

```text
180,115,44,158,114,3,182,195
```

출력은 `outputs/tree_<tag>.jsonl`에 기록된다. 이 스크립트는 partial prefix만 생성하며 Base
완결문, Judge label, Qwen3-14B validation, correctness를 만들지 않는다.

### Tree JSONL

| 필드 | 내용 |
|---|---|
| `problem_id` | 문제 ID |
| `model` | Base 모델 경로 |
| `threshold` | candidate probability cutoff |
| `entropy_threshold` | internal fork entropy cutoff |
| `max_depth` | opening을 포함한 최대 branch depth |
| `max_follow` | 다음 fork를 찾으며 top-1으로 걷는 최대 token 수 |
| `path` | forced branch token과 Base 확률 |
| `path_prob` | forced·greedy token을 모두 포함한 prefix 확률 |
| `depth` | leaf가 끝난 depth |
| `walked` | 마지막 fork 뒤에서 top-1으로 걸은 token 수 |
| `end_reason` | `max_depth`, `max_follow`, `no_candidates` |
| `end_entropy` | leaf 끝 위치의 entropy |
| `leaf_text` | prompt 뒤의 partial prefix text |
| `token_ids` | partial prefix token IDs |

## 주요 스크립트

| 파일 | 용도 | 상태 |
|---|---|---|
| `select_problems.py` | 20개 discovery 문제 선정 | 완료 |
| `run_discovery.py` | 6개 API source 병렬 호출 | 256-token pilot용 |
| `gen_habitat.py` | Base 자연 rollout temperature sweep | 완료 |
| `read_habitat.py` | Base blind label 집계 | 완료 |
| `branch_tree.py` | Base recursive partial-prefix tree | 현재 생성기 |
| `embed_only.py` | discovery raw embedding 생성 | 진단용 |
| `diag_style_axis.py` | source/style 편향 검사 | 완료된 진단 |
| `prefill_score.py` | 고정 trajectory log-prob 채점 | 이전 pilot |

Base leaf validation과 최종 panel hard classification은 별도 스크립트로 구현한다. Validation은
leaf당 Qwen3-14B continuation 네 개를 만들고, 기존 수학 채점과 Judge의 approach-preservation
판정을 함께 저장해야 한다.

## 주요 산출물

| 경로 | 내용 |
|---|---|
| `outputs/discovery_pilot_v2.jsonl` | API partial 요청 1,920건, 성공 1,913건 |
| `outputs/master_traj.jsonl` | API trajectory, label, prefill 점수 결합본 |
| `outputs/labels_json/` | API trajectory의 문제별 approach label |
| `outputs/habitat_base_T3.jsonl` | Base T=0.7/1.0/1.3 rollout |
| `outputs/label_base/` | Base T=0.7/1.0 blind label |
| `outputs/tree_smoke.jsonl` | p180 opening-tree pilot 35 leaf |
| `outputs/tree_<tag>.jsonl` | recursive Base tree 출력 |
| `outputs/diag_style_axis.txt` | raw embedding의 source 편향 진단 |
| `figures/` | embedding 진단 그림 |

API 생성물은 다시 만들면 비용이 든다. Raw JSONL과 label은 보존한다. `emb_*.npy`는 로컬
embedding 모델로 다시 만들 수 있으므로 Git에서 제외한다.

## 경로 주의

일부 초기 pilot 스크립트에는 이전 작업 경로인 `WorkPlace.ICLR` import가 남아 있다. 현재 정본
경로는 `/home/eoeldroal/WorkPlace/Distill_start`다. 새 실행에는 `or_common.py`와
`branch_tree.py`처럼 현재 디렉터리를 기준으로 경로를 계산하는 스크립트를 사용한다.
