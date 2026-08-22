# 실험 설계

이 문서는 프로젝트 전체에 걸쳐 확정된 실험값과 비교 arm을 기록한다. Branch panel과 \(E\)의 상세 프로토콜은 [Branch Panel and E](Branch_Panel_and_E.md)를 정본으로 삼고, 여기서는 다른 실험과 맞물리는 값만 요약한다. \(\beta\) 설계 수치의 출처 스크립트는 `toy_sims/`에 있다.

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

**Branch panel.** Branch identity는 문제별 primary mathematical approach로 고정한다. Judge가 hard classification하며, 원문 embedding은 branch 배정에 쓰지 않는다. API discovery는 서로 다른 여섯 모델에서 문제당 완결 풀이 16개씩 모으고, Base discovery는 재귀 tree로 partial prefix를 찾는다.

| Base tree 항목 | 고정값 |
|---|---:|
| opening 전개 | 첫 두 token, 각 token probability \(\ge 1\%\) |
| internal fork | entropy \(\ge 1.5\), 후보 둘 이상이 각각 \(\ge 1\%\) |
| fork 사이 진행 | greedy/top-1 |
| 최대 깊이 | 3 |
| 다음 fork 탐색 상한 | 48 token |
| leaf validation | Qwen3-14B, K=4 |
| 수용 기준 | 정답이면서 접근법을 유지한 완결이 한 번 이상 |

API 정답 완결문과 검증된 Base leaf를 합쳐 panel을 만들고, 이후 모든 checkpoint에 같은 taxonomy를 적용한다. 생성 기반 \(E^{\mathrm{gen}}\)은 모든 응답에 대한 hard frequency이며, prefill 기반 \(E^{\mathrm{entry}}\)는 Base에서 검증된 고정 entry prefix의 covered mass다. API-only branch의 representative entry는 실제 panel을 본 뒤 정한다. API endpoint와 실행 명령은 [BranchDev README](../Experiment/BranchDev/README.md)를 따른다.

기존 temperature 1.3, max_tokens 256의 API pilot은 1,920건을 요청해 1,913개의 성공 trajectory를 얻었다. 이 자료와 문제 180의 35-leaf tree는 설계를 결정한 pilot evidence로만 보존한다. 최종 panel 입력은 위 완결 풀이와 검증 절차를 새로 적용해 만든다.

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

m_min은 보호 효과를 분석할 최소 Base branch mass이자 무릎 공식의 입력이다. Panel membership이나 tree의 token cutoff로 사용하지 않는다. 값은 run-level 95%, N=80 기준으로 네 제약에서 정한다.

1. 구조 가능성: β ≤ 1이므로 m ≥ 1 − 0.05^(1/N) ≈ 0.037. 이 밑은 완전 보존으로도 목표 달성이 불가능하다.
2. 측정 가능성: E 추정은 Monte Carlo이므로 m ≳ 15/M. State당 rollout M=256이면 약 0.06. 측정 못 하는 branch에 대한 약속은 검증 불능이다.
3. β 비용: β ≤ 0.5로 두려면 m ≥ 0.037/0.5 ≈ 0.074.
4. 현상 서식지: distillation이 짓누르는 소수 branch는 anchor mass 0.05~0.2 구간에 산다. m_min이 0.15를 넘으면 보호 대상이 소멸한다.

교집합은 대략 [0.074, 0.15]이고 중앙값 0.10을 채택한다. 이 값은 panel 구축 뒤 실제 Base-validated branch mass에 대한 sensitivity analysis로 다시 확인한다.

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
| 입력 | m_min | 0.10 | 네 제약의 교집합 중앙 (§1.3), 보호 효과 분석 기준 |
| 산수 | N, β_knee, 커버리지 함수 | 80, 0.37, 0.037/β | §1.2 공식 |
| 실측 예정 | k_bind | 진입 창의 binding 수 | teacher와 anchor의 forward pass만으로 계산 가능 |
| 실측 예정 | T_eff | 잠정 5 | RL 중간 checkpoint의 E 궤적 |
| 실측 예정 | V_j | branch별 | anchor rollout |

### 1.6 retention의 정밀화

보장 사슬은 token floor에서 branch 진입 회복으로 간다. 진입이 token 하나로 결정되면 retention은 β이고, 일반적으로는 binding 위치 수 k_bind에 대해 β^k_bind다. binding 여부는 (π_T, π_A, s)만의 성질이라 student와 무관하고, 따라서 훈련 전에 계산할 수 있다. β^k_bind는 설계용 산수이며, 검증은 panel에 고정한 entry prefix의 student/anchor probability ratio로 한다. k_bind는 접근법 진입이 확인되는 prefix 구간 안에서만 세고, 이후 token의 binding은 branch 내부 실행 보존을 보는 별도 진단으로 남긴다.

## 2. 확정된 실험 목록

1. 주 비교: Anchor → Vanilla OPD → RL vs Anchor → Protected OPD(β=0.4) → RL. 연구 본문의 기본 설계.
2. β sweep {0.1, 0.2, 0.4, 0.8} + vanilla, 전 arm distillation과 RL 수행 (§0).
3. 순방향 KL divergence baseline과의 비교. λ sweep으로 수행하며 distillation과 handoff 측정만 필요하다 (RL 없음).
4. Arithmetic mixture baseline: 단일점 α = β = 0.4, distillation과 RL 수행.
   sweep을 하지 않는 근거: mixture의 내장 floor는 정확히 α·π_A이므로 α = β가 canonical projection과 보장 수준을 정확히 일치시킨다 (matched guarantee). α는 우리가 고른 값이 아니라 matching 규칙이 정한 값이라 "불리한 α" 공격이 성립하지 않고, 다른 α는 최적성 정리가 커버한다 (임의 α의 mixture는 β=α projection에 지배됨). Penalty는 실효 보호가 불균일해 이런 matching이 불가능하므로 λ sweep을 유지한다.
   RL을 붙이는 근거: matched guarantee에서 mixture가 낮은 pass@1로 출발해 post-RL이 projection과 같거나 좋다면, soft 영역의 분포 평탄함 자체가 RL에 중요하다는 경쟁 가설의 증거가 된다. 이 arm은 우리 방법이 질 수 있는 진짜 경쟁 가설을 시험하는 자리다.
5. Recovery 분석의 기본 intervention 2종: 추가 RL compute(기존 run 연장)와 rollout temperature 조절(추가 학습 없음). 더 비싼 intervention(entropy-controlled RL, self-reheating/self-distillation, anchor repair)은 이 둘로 복구되지 않는 손상이 확인될 때만 확장한다. §6 prevention-repair 비교의 repair는 기본 2종을 사용한다.

## 3. 본문 정의와의 정합

`Draft.md`와 Branch/E 정본은 다음 정의를 공유한다.

1. Branch는 token이나 문체가 아니라 문제별 primary mathematical approach다.
2. API 완결 풀이와 Base partial-prefix tree는 서로 다른 discovery 역할을 맡지만, 최종 panel에서는 같은 judge taxonomy로 병합한다.
3. Base leaf는 Qwen3-14B K=4 validation에서 정답이면서 접근법을 유지한 완결이 한 번 이상 나올 때만 entry prefix로 채택한다.
4. Judge는 panel 구축과 checkpoint 응답 배정 모두에서 hard classification을 수행한다. `other`, `ambiguous`, `failed`는 별도 범주다.
5. \(E^{\mathrm{gen}}\)은 실제 생성 빈도, \(E^{\mathrm{entry}}\)는 Base에서 검증된 고정 entry prefix가 포괄하는 probability mass다. 후자를 branch 전체 확률로 해석하지 않으며, API-only entry 구성은 panel을 본 뒤 정한다.
6. 원문 embedding은 시각화에만 사용하며 branch identity나 배정을 결정하지 않는다.
7. Base에서 검증된 branch와 API에서만 발견된 branch를 구분한다. 후자는 panel coverage에는 포함하지만 Base가 잃은 경로라는 주장에는 쓰지 않는다.

세부 정의와 실행 순서는 [Branch Panel and E](Branch_Panel_and_E.md)를 따른다.

## 4. 확정된 분석 목록

논증으로 이미 종결된 결과(최적성 부등식, mixture의 floor 내장, 닫힌 해와 odds 보존, 역방향 KL의 소멸 비용, 무릎 공식)는 본문과 appendix의 내용이며 이 목록의 대상이 아니다. 실험은 새 데이터가 필요한 질문에, 분석은 이미 있는 숫자로 답할 수 있는 질문에 쓴다.

### 4.1 사전 분석과 panel 구축

- Cost(β) 예산표: q*가 닫힌 형태이므로 state 표본에서 KL(q*‖π_T)를 직접 계산한다. β 선택의 비용 쪽 근거이며, [Cal Beta Before Train](Cal_Beta_Before_train.md)에 정리되어 있다.
- k_bind와 binding profile: binding은 (π_T, π_A, s)만의 성질이다. Panel에서 고정한 Base entry prefix를 측정 창으로 삼고, 접근법 진입 전 구간의 binding만 k_bind에 센다.
- Pilot evidence: API 요청 1,920건 중 성공한 partial trajectory 1,913개, raw embedding의 source bias, Base approach habitat, 문제 180의 35-leaf tree는 설계 근거로 이미 확보했다. 결과와 현재 해석은 [Branch Panel and E §8](Branch_Panel_and_E.md#8-기존-실측이-남긴-결정)에 보존한다.
- Final panel 구축: API 여섯 모델의 완결 풀이와 Base recursive tree를 생성하고, Qwen3-14B K=4 validation과 judge hard classification을 거쳐 문제별 taxonomy와 validated Base entry prefix를 동결한다.
- Checkpoint 측정 준비: 모든 rollout의 원문, 정답 여부, hard branch label을 보존하고, 같은 panel에서 \(E^{\mathrm{gen}}\)과 \(E^{\mathrm{entry}}\)를 함께 계산한다.

### 4.2 사후 분석 (실험 산출물 위의 계산, 새 run 불필요)

- predictor 분석: held-out loss와 Pass@large-k를 이미 아는 상태에서 D_succ(G)가 post-RL 결과를 추가로 설명하는지. 분석 단위는 recipe × seed.
- 무릎 검증: 사전 분석의 예측(비용, retention)과 handoff 실측(pass@1, \(E^{\mathrm{entry}}\), \(E^{\mathrm{gen}}\))의 대조. post-RL 성능이 β=0.4 근방에서 포화하는지.
- G-sensitivity: 같은 panel 측정치에서 D_succ(G)를 G ∈ {8, 16, 32}로 재계산.
- m_min sensitivity: E 측정 원자료에 문턱 {0.05, 0.10, 0.15}를 재적용.
- 반비례 진단: λ arm들에서 실효 보호를 teacher crush 강도별로 binning.
- T_eff 추정: RL 중간 checkpoint의 E 궤적에서 미발견 branch가 눌리는 속도를 읽는다.
- RL 학습 신호: GRPO 로그에서 all-fail, all-success, informative group 비율.
- 기제 사슬 정렬: floor binding → \(\Delta E^{\mathrm{entry}}\) → \(\Delta E^{\mathrm{gen}}\) → D_succ → informative group (Draft §3.6).
- frontier figure: 각 arm의 (handoff 보호, handoff pass@1)을 한 평면에 그린다. β arm들이 projection 곡선, λ arm들이 penalty 곡선, mixture arm이 한 점.

### 4.3 사후 분석이 요구하는 로깅

- RL 중간 checkpoint 저장과 주기적 panel 측정 (T_eff와 궤적 추적)
- GRPO group별 reward 로그 (informative 비율)
- checkpoint별 held-out loss와 Pass@large-k (predictor baseline)
- panel 측정 원자료를 집계 전 상태로 보존: rollout별 branch 배정 (G와 m_min 재계산)
- floor arm과 λ arm에서 state 표본의 (π_A, π_T, target) 저장 (binding profile과 반비례 진단)
