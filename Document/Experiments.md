# 실험 설계

이 문서는 합의가 끝난 설계 결정과 그 도출 과정만 기록한다. 아직 논의 중인 항목은 §5에 이름만 적어 두고, 결정으로 취급하지 않는다. 수치의 출처 스크립트는 `toy_sims/`에 있다.

## 0. 확정된 설계 결정

**Sampler.** Canonical sampler는 full softmax로 한다. top-p, top-k를 적용하지 않으며, 실제 RL rollout도 같은 설정으로 돌린다. Temperature는 측정과 RL에서 동일하게 고정한다.
근거: rollout 분포가 정책 분포와 일치해 policy gradient가 순수 on-policy가 된다. 보장 사슬에서 truncation 생존 가정이 통째로 사라져 수학이 단순해진다.
파급: hard entry loss의 정의가 budget 기반 하나로 통일된다 (진입 확률이 너무 작아 유효 budget 안에서 사실상 관찰되지 않는 경우). 본문 정의 수정은 §3.

**G = 16.** GRPO 그룹 크기. G=8 대비 필요 floor가 절반으로 내려가고, informative group 비율에도 유리하다. 관행 범위(8~64) 안이다.

**Canonical β = 0.4.** 도출 과정은 §1. 심층 분석(matched 비교, 기제 분석)은 이 arm을 기준으로 한다.

**β sweep = {0.1, 0.2, 0.4, 0.8}, 모든 arm에 RL 수행.** vanilla(β=0)는 연구 본문의 주 비교(Anchor → Vanilla OPD → RL)가 이미 요구하는 대조군이다. 전 arm에 RL을 도는 이유: post-RL 결과가 이 연구의 종속변수이므로, RL 없는 arm은 예측자 분석에 기여하지 못한다. 5개 arm이 (handoff D_succ, post-RL 성능) 산점도의 점이 되고, 단조 dose-response 추세 자체가 분산에 대한 방어를 겸한다.

Sweep grid의 좌표 번역 (N = G×T_eff = 80, run-level 95% 기준, toy 수치는 `toy_sims/beta_design.py`):

| β | 포화 커버리지 (m ≥) | 서식지 branch run-level (m=0.15) | toy 비용 (KL) |
|---|---|---|---|
| 0 (vanilla) | 없음 | 0.27 | 0 |
| 0.1 | 0.37 | 0.70 | 0.004 |
| 0.2 | 0.18 | 0.91 | 0.026 |
| 0.4 | 0.09 | 0.99 | 0.10 |
| 0.8 | 0.05 | 1.00 | 0.37 |

0.1과 0.2는 dose-response 곡선의 가파른 구간, 0.4는 무릎, 0.8은 무릎 너머다. 0.8은 무릎 이론의 반증 가능한 예측을 시험한다: D_succ가 0.4와 같고 비용만 크므로, post-RL이 같거나 나빠야 이론이 맞다.

**Discovery 소스 6개.** GLM 5.2, DeepSeek V4 Flash 0731, Qwen3.8-27B, MiniMax M3, MiMo v2.5 Pro, Muse Glimmer 30B를 OpenRouter로 호출하고, 모델마다 endpoint를 고정한다. 세 조건을 실측으로 확인해 고른 것이다: rate limit 없음, reasoning 평문 반환, Artificial Analysis Intelligence Index가 신뢰 하한 이상. AAII는 하한으로만 쓰고 그 위에서 줄 세우지 않는다. 실측에서 AAII와 접근 다양성은 역상관이었다.
근거: 다양성의 주 동력은 모델 간 차이다. 같은 모델에서 표본을 늘리는 것으로는 접근이 갈리지 않고, 온도로 갈리는 것처럼 보이는 구간에서는 텍스트가 무너진다. 상세는 Cal_E_Before_train.md §2.

**Discovery 생성 설정.** temperature 1.3, top_p 1.0, top_k 0, max_tokens 256, reasoning 노출. endpoint 고정에 `allow_fallbacks: false`를 건다.
근거: endpoint가 top_p나 top_k를 선언 지원하지 않으면 값을 보내도 오류 없이 버리므로, 고정하지 않으면 sampling 설정이 호출마다 달라진다. max_tokens 256은 reasoning과 답변의 합에 걸려 실질적으로 사고의 앞 256 token이 되는데, 넉넉히 받아 나중에 자르는 방식과 앞부분 텍스트가 통계적으로 동일하고 비용은 절반이다.

**프롬프트 지시문의 공유.** `"<문제>\n\nPlease reason step by step, and put your final answer within \boxed{}."` 이 문자열을 사전 분석, discovery, E와 V 측정, RL rollout이 전부 공유한다.
근거: cluster를 만든 텍스트와 거기 배정될 rollout의 프롬프트가 같아야 배정이 성립한다. 이 지시문은 벤치마크 프로토콜이 아니라 V-E 측정 전용이므로, 이 설정에서 나온 수치로 벤치마크 성능을 주장하지 않는다.

주의: sampler와 G 결정은 둘 다 vanilla distillation의 해악을 완화하는 방향이다. 이 관대한 세팅에서 효과가 나오면 주장이 강해지고, 안 나오면 현상이 truncation 의존적이었다는 것 자체가 발견이다.

## 1. 파라미터 결정 과정 (논문 appendix 초안)

### 1.1 원리: 최적 β는 내부에 없고 min(무릎, 예산)에 있다

발견 이득과 fidelity 비용이 모두 β에 단조 증가하므로 내부 최적점이 존재하지 않는다. 따라서

    β* = min( β_knee, Cost⁻¹(δ) )

β_knee는 이득이 포화하는 지점, Cost⁻¹(δ)는 fidelity 예산 δ가 허용하는 최대 보호다. δ는 새 상수가 아니라 matched-distillation protocol의 pass@1 허용 오차를 상속한다. Cost(β) = 평균 KL(q*‖π_T)는 q*가 닫힌 형태이므로 forward pass만으로, 훈련 없이 계산된다.

### 1.2 무릎 공식

완전히 짓눌린 branch(anchor mass m)의 floor 후 진입 확률은 β·m이고, 유효 시도 N = G×T_eff에 대해

    P_run = 1 − (1 − β·m)^N ≈ 1 − exp(−β·m·N)

한계이득이 지수 감쇠하므로 무릎이 뚜렷하다. run-level 신뢰도 c에서

    β_knee(m) = ln(1/(1−c)) / (m·N)        (c=95%이면 ln 20 ≈ 3)

상수 3조차 임의값이 아니라 신뢰도의 함수다.

### 1.3 m_min: 네 제약의 교집합

m_min은 보호를 약속하는 최소 anchor mass다. 본문 §3.1에서 inherited branch 자격을 정하는 최소 anchor mass 기준과 같은 상수이며, panel의 자격선과 무릎 공식의 입력이라는 두 역할에 하나의 값을 쓴다. 네 방향에서 죄인다. 값은 run-level 95%, N=80 기준.

1. 구조 가능성: β ≤ 1이므로 m ≥ 1 − 0.05^(1/N) ≈ 0.037. 이 밑은 완전 보존으로도 목표 달성이 불가능하다.
2. 측정 가능성: E 추정은 Monte Carlo이므로 m ≳ 15/M. State당 rollout M=256이면 약 0.06. 측정 못 하는 branch에 대한 약속은 검증 불능이다.
3. β 비용: β ≤ 0.5로 두려면 m ≥ 0.037/0.5 ≈ 0.074.
4. 현상 서식지: distillation이 짓누르는 소수 branch는 anchor mass 0.05~0.2 구간에 산다. m_min이 0.15를 넘으면 보호 대상이 소멸한다.

교집합은 대략 [0.074, 0.15]이고 중앙값 0.10을 채택한다. 최종 고정은 branch-dev에서 한다.

### 1.4 도출

T_eff = 5 (보수적 잠정치; crowding-out 경쟁을 존중해 작게 잡음, RL 중간 checkpoint의 E 궤적에서 추후 실측), N = 16×5 = 80, m_min = 0.10:

    β_knee = 0.037 / 0.10 ≈ 0.37  →  canonical β = 0.4

이 β에서 toy 비용은 KL ≈ 0.10 nats이고, anchor mass 0.09 이상의 branch는 run-level 발견 95% 이상이 예측된다. T_eff를 보수적으로 잡았으므로 오차는 과보호 쪽으로 나며, sweep이 아래 방향으로 교정할 수 있다.

### 1.5 상수의 신분표

| 구분 | 항목 | 값 | 출처 |
|---|---|---|---|
| 선언 | run-level 신뢰도 | 95% | 관행적 신뢰수준 |
| 선언 | fidelity 예산 δ | matched protocol 허용 오차 | 기존 프로토콜 상속 |
| 입력 | G | 16 | RL 설정 (§0) |
| 입력 | M (panel rollout 수) | 측정 예산에서 | 제약 2의 하한 결정 |
| 입력 | m_min | 0.10 | 네 제약의 교집합 중앙 (§1.3), 본문 §3.1의 inherited 자격선과 동일 상수 |
| 산수 | N, β_knee, 커버리지 함수 | 80, 0.37, 0.037/β | §1.2 공식 |
| 실측 예정 | k_bind | 진입 창의 binding 수 | teacher와 anchor의 forward pass만으로 계산 가능 |
| 실측 예정 | T_eff | 잠정 5 | RL 중간 checkpoint의 E 궤적 |
| 실측 예정 | V_j | branch별 | anchor rollout |

### 1.6 retention의 정밀화

보장 사슬은 token floor에서 branch 진입 회복으로 간다. 진입이 token 하나로 결정되면 retention은 β이고, 일반적으로는 binding 위치 수 k_bind에 대해 β^k_bind다. binding 여부는 (π_T, π_A, s)만의 성질이라 student와 무관하고, 따라서 훈련 전에 계산할 수 있다. β^k_bind는 설계용 산수이며, 검증은 segment 수준 직측(진입 확률의 student/anchor 비)으로 한다. binding 측정 창은 branch clustering에 쓰는 창(32~64 token)과 통일하고, k_bind에는 창 앞부분(진입이 확정되기 전 구간)의 binding만 센다. 창 뒷부분의 binding은 진입이 아니라 branch 내부 실행(V) 보존의 진단에 쓴다.

## 2. 확정된 실험 목록

1. 주 비교: Anchor → Vanilla OPD → RL vs Anchor → Protected OPD(β=0.4) → RL. 연구 본문의 기본 설계.
2. β sweep {0.1, 0.2, 0.4, 0.8} + vanilla, 전 arm distillation과 RL 수행 (§0).
3. 순방향 KL divergence baseline과의 비교. λ sweep으로 수행하며 distillation과 handoff 측정만 필요하다 (RL 없음).
4. Arithmetic mixture baseline: 단일점 α = β = 0.4, distillation과 RL 수행.
   sweep을 하지 않는 근거: mixture의 내장 floor는 정확히 α·π_A이므로 α = β가 canonical projection과 보장 수준을 정확히 일치시킨다 (matched guarantee). α는 우리가 고른 값이 아니라 matching 규칙이 정한 값이라 "불리한 α" 공격이 성립하지 않고, 다른 α는 최적성 정리가 커버한다 (임의 α의 mixture는 β=α projection에 지배됨). Penalty는 실효 보호가 불균일해 이런 matching이 불가능하므로 λ sweep을 유지한다.
   RL을 붙이는 근거: matched guarantee에서 mixture가 낮은 pass@1로 출발해 post-RL이 projection과 같거나 좋다면, soft 영역의 분포 평탄함 자체가 RL에 중요하다는 경쟁 가설의 증거가 된다. 이 arm은 우리 방법이 질 수 있는 진짜 경쟁 가설을 시험하는 자리다.
5. Recovery 분석의 기본 intervention 2종: 추가 RL compute(기존 run 연장)와 rollout temperature 조절(추가 학습 없음). 더 비싼 intervention(entropy-controlled RL, self-reheating/self-distillation, anchor repair)은 이 둘로 복구되지 않는 손상이 확인될 때만 확장한다. §6 prevention-repair 비교의 repair는 기본 2종을 사용한다.

## 3. 본문 정의와의 정합

반영 완료. 아래는 Draft.md의 현재 상태다.

1. Floor의 적용 범위는 전체 vocabulary다: q(v) ≥ β·π_A(v), 모든 v. 보호 크기가 anchor 확률에 비례하므로 조준이 내장되어 있고, β < 1이면 제약은 항상 실현 가능하다 (§5.1).
2. hard entry loss는 budget 기준 하나로 정의한다: 진입 확률이 너무 작아 실제 rollout budget 안에서 사실상 관찰되지 않는 경우 (§3.3).
3. Canonical sampler는 truncation 없는 full softmax, temperature는 RL rollout과 동일 (§3.1).
4. Binding 측정 창은 branch clustering의 continuation 길이와 같고, 창 앞부분의 binding 수만 진입 확률 환산에 쓴다 (§3.6).
5. 방법의 이름은 relative-floor projection이다 (§5.1).
6. Prefill 채점 L_θ = log P_θ(r|s)를 E의 보완 측정으로 병기한다. E의 정의는 생성 기반 그대로 두고, L_θ는 그 하한으로 명시한다 (§3.3).
7. L_θ는 raw log-probability로 다루고 정규화하지 않으며, 서로 다른 continuation의 값을 직접 비교하지 않는다. 측정에 쓰는 양은 같은 continuation에 대한 두 checkpoint의 차이다 (§3.3).
8. Discovery 소스의 문체 이질성은 줄이지 않되, checkpoint 비교 결과는 소스별로 층화해 보고한다 (§3.1).
9. LLM-as-a-judge는 branch assignment에 쓰지 않는다. embedding 공간을 검증하기 위한 독립 기준을 만드는 데만 쓰며, 이때 판정자는 블라인드이고 문제별 고정 label 목록에서만 고르며 문체를 판단 근거로 삼지 않는다 (§3.1).

## 4. 확정된 분석 목록

논증으로 이미 종결된 결과(최적성 부등식, mixture의 floor 내장, 닫힌 해와 odds 보존, 역방향 KL의 소멸 비용, 무릎 공식)는 본문과 appendix의 내용이며 이 목록의 대상이 아니다. 실험은 새 데이터가 필요한 질문에, 분석은 이미 있는 숫자로 답할 수 있는 질문에 쓴다.

### 4.1 사전 분석 (훈련 전, frozen 모델의 forward pass만 필요)

- Cost(β) 예산표: q*가 닫힌 형태이므로 state 표본에서 KL(q*‖π_T)를 직접 계산한다. β 선택의 비용 쪽 근거. 수행 완료, Cal_Beta_Before_train.md.
- k_bind와 binding 프로파일: binding은 (π_T, π_A, s)만의 성질이다. inherited branch의 진입 창(branch clustering과 동일한 32~64 token)에서 위치별 binding을 기록하고, 창 앞부분의 개수를 k_bind로 쓴다. 수행 완료, Cal_Beta_Before_train.md §7.
- embedding 공간의 수용 검사: discovery traj에 독립 approach label을 붙이고, 방법만 공유하는 쌍과 소스만 공유하는 쌍의 거리를 비교한다. panel을 짓기 전에 좌표계가 방법을 보는지 문체를 보는지 판정한다. 수행 완료, Cal_E_Before_train.md §4.
- prefill 채점 L_θ: 고정된 traj에 checkpoint의 log P를 매긴다. 배정 단계를 거치지 않아 문체가 개입할 자리가 없고, 보장 q* ≥ β·π_A와 같은 확률 단위로 검증한다. E의 하한이라는 점을 함께 보고한다. 수행 완료, Cal_E_Before_train.md §5.

### 4.2 사후 분석 (실험 산출물 위의 계산, 새 run 불필요)

- predictor 분석: held-out loss와 Pass@large-k를 이미 아는 상태에서 D_succ(G)가 post-RL 결과를 추가로 설명하는지. 분석 단위는 recipe × seed.
- 무릎 검증: 사전 분석의 예측(비용, retention)과 handoff 실측(pass@1, E 회복)의 대조. post-RL 성능이 β=0.4 근방에서 포화하는지.
- G-sensitivity: 같은 panel 측정치에서 D_succ(G)를 G ∈ {8, 16, 32}로 재계산.
- m_min sensitivity: E 측정 원자료에 문턱 {0.05, 0.10, 0.15}를 재적용.
- 반비례 진단: λ arm들에서 실효 보호를 teacher crush 강도별로 binning.
- T_eff 추정: RL 중간 checkpoint의 E 궤적에서 미발견 branch가 눌리는 속도를 읽는다.
- RL 학습 신호: GRPO 로그에서 all-fail, all-success, informative group 비율.
- 기제 사슬 정렬: floor binding → entry lift → ΔE_j → D_succ → informative group (§3.6).
- frontier figure: 각 arm의 (handoff 보호, handoff pass@1)을 한 평면에 그린다. β arm들이 projection 곡선, λ arm들이 penalty 곡선, mixture arm이 한 점.

### 4.3 사후 분석이 요구하는 로깅

- RL 중간 checkpoint 저장과 주기적 panel 측정 (T_eff와 궤적 추적)
- GRPO group별 reward 로그 (informative 비율)
- checkpoint별 held-out loss와 Pass@large-k (predictor baseline)
- panel 측정 원자료를 집계 전 상태로 보존: rollout별 branch 배정 (G와 m_min 재계산)
- floor arm과 λ arm에서 state 표본의 (π_A, π_T, target) 저장 (binding 프로파일과 반비례 진단)

## 5. 논의 대기 (합의되지 않음, 결정 아님)

- seed 반복 횟수와 RL run 간 분산의 잣대
- arm별 측정 깊이의 차등 (전 궤적 추적 vs handoff와 최종만)
- matched protocol을 sweep 전체에 어떻게 적용할지
- top-p = 0.95 ablation: 새 RL run이 필요한 유일한 보조 실험 후보
- 역방향 KL 데모: 소규모 distillation 후보
- 진입 확정 길이: branch의 정체가 몇 token 만에 확정되는지. 무릎 β_knee가 이 값으로 β^k 재계산되므로 결정 하나가 여기에 걸려 있다 (Cal_Beta_Before_train.md §7.3)
- 문제 선별 기준: 난이도로는 접근 다양성을 예측할 수 없다는 것까지 확인됐고, 조작적 정의가 없다 (Cal_E_Before_train.md §7)
