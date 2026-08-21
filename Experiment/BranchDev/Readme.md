# Branch discovery 생성 설정

## 요청 본문

```json
{
  "model": "<소스>",
  "messages": [{"role": "user",
    "content": "<문제>\n\nPlease reason step by step, and put your final answer within \\boxed{}."}],
  "temperature": 1.3,
  "top_p": 1.0,
  "top_k": 0,
  "max_tokens": 256,
  "reasoning": {"enabled": true, "exclude": false},
  "transforms": [],
  "provider": {"order": ["<고정 endpoint>"], "allow_fallbacks": false}
}
```

`seed`와 `reasoning.effort`는 넣지 않는다.

## 소스

| 모델 | endpoint | AAII |
|---|---|---|
| GLM 5.2 | `ambient/fp8` | 52.6 |
| DeepSeek V4 Flash 0731 | `deepinfra/fp8` | 51.8 |
| Qwen3.8-27B | `chutes/fp8` | 52 |
| MiniMax M3 | `deepinfra/fp8` | 45.4 |
| MiMo v2.5 Pro | `deepinfra/fp8` | 42.9 |
| Muse Glimmer 30B | `deepinfra/bf16` | 35 |

전부 rate limit 없음, T=1.3 수용, reasoning 평문 반환, 세 sampling 파라미터 지원을 실측으로 확인했다.
여섯 가문이며, 다양성의 주 동력은 모델 간 차이다.

AAII는 하한으로만 쓰고 그 위에서 줄 세우지 않는다. 실측에서 AAII와 접근 다양성은 역상관이었다.
AAII 51.8인 DeepSeek이 문제당 평균 2.30가지로 가장 좁았고, 45.4인 MiniMax와 42.9인 MiMo가
각각 3.65와 4.20으로 가장 넓었다.

## 실수하면 안 되는 것

**endpoint를 고정하고, 그 endpoint가 `top_p`/`top_k`를 선언 지원하는지 확인한다.** 지원하지 않는
endpoint는 값을 보내도 오류 없이 버린다. 응답의 `provider` 필드를 기록해 고정이 유지되었는지
사후 확인한다.

**rate limit은 모델 단위이고 정각 분당으로 리셋된다.** 동시성 제한이 아니므로 동시성을 낮춰도
풀리지 않고, 같은 모델의 다른 endpoint로 바꿔도 버킷을 공유한다. 크레딧 충전으로도 풀리지 않는다.
위 다섯 모델은 이 버킷이 없는 것을 확인했다.

**`max_tokens`는 reasoning과 content의 합에 걸린다.** reasoning이 먼저 생성되므로 256이면
사고의 앞 256 토큰이 오고 `content`는 대개 빈다. discovery에는 정답이 필요 없으므로 문제가
아니다(V는 checkpoint 자신의 rollout에서 잰다).

**프롬프트 지시문은 사전 분석·discovery·E/V 측정·RL rollout이 전부 공유한다.** cluster를 만든
텍스트와 거기 배정될 rollout의 프롬프트가 같아야 한다.

**traj 사이의 절대 비교를 하지 않는다.** 서로 다른 traj의 확률이나 하락폭을 나란히 놓는 것은 진단
목적으로만 쓴다. 측정에 쓰는 양은 언제나 같은 traj에 대한 두 checkpoint의 차이다. 그 차이에서만
traj 고유의 문체와 난이도가 소거된다. 근거는 Cal_E_Before_train.md §4다.

**결과는 소스별로 층화해서 보고한다.** 여섯 소스 각각에서 arm 차이를 내고 나란히 제시한다.
층화하면 문체 효과가 층 안에 갇히고 특정 소스의 아티팩트라는 반론이 막힌다.

**임베딩 시각화에는 PCA를 쓰고 t-SNE/UMAP을 쓰지 않는다.** 우리 주장은 거리와 분리도에
대한 것인데 두 기법은 국소 이웃만 보존하고 군집 간 거리에 의미가 없다. PCA는 파라미터가
없어 조작 여지도 없다. 투영은 문제 안에서만 하고(전체로 하면 첫 축이 "어느 문제인가"를
잡는다), 두 축의 분산 비율과 trustworthiness/continuity를 함께 보고한다. 그림은 원공간에서
계산한 통계를 대신하지 않고 보여 주기만 한다. `plot_embedding.py`가 이를 구현한다.

**절대 성능을 주장하지 않는다.** "이 방법으로 MATH 점수가 올랐다"는 형태로 쓰지 않는다.
handoff pass@1은 matched protocol이 요구하므로 측정하되, 두 arm이 같은 출발점에 있었다는
증거로만 제시한다.

## 알아 두면 좋은 것

온도는 다경로 문제에서만 효과가 있다. 문제 180(Level 5 대수)에서 DeepSeek이 T=1.0에서
3종/87%였다가 T=1.3에서 7종/50%로 갈렸다. 경로가 하나뿐인 문제에서는 T를 올려도 어휘만
바뀐다. T=1.6 이상에서는 텍스트가 깨지기 시작한다.

난이도로는 다양성을 예측할 수 없다. Level 5인 180번이 Level 2인 3번만큼 균일한 경우가 있다.

API 모델의 양자화는 통제할 수 없다(fp4/fp8/unknown). 텍스트만 쓰므로 무해하며, 확률을 읽는
계산은 전부 로컬 모델에서 full precision으로 한다.

## 미결

- 진입 확정 길이 — P_run이 78%에서 0.9%까지 갈린다 (Cal_Beta_Before_train.md §7.3)
- 문제 선별 기준
- checkpoint 간 문체 격차 — 사전 분석의 anchor rollout 800개와 teacher rollout 200개로 확인 가능
- 생성 기반 E와 prefill의 대조
