# Distillation → RL 인계의 탐색 구조와 회복 가능성

## 요약

최근 frontier model의 post-training은 SFT, RL, distillation을 한 번씩 거치는 선형 pipeline이 아니다. 여러 training stage를 오가며 capability를 만들고, 합치고, 다시 강화하는 과정에 가깝다. 이 과정에서 on-policy distillation(OPD)은 단순한 model compression을 넘어, 서로 다른 RL specialist의 능력을 하나의 policy로 통합하거나 stage 사이에서 capability를 옮기는 수단으로 쓰이기 시작했다.

DeepSeek-V4, Kimi K3, GLM-5는 OPD 또는 cross-stage distillation을 대규모 capability consolidation에 활용한다. Nemotron-Cascade 2는 OPD를 RL stage 사이의 중간 단계로 배치하고, Qwen3.5-Omni도 OPD 이후 다시 RL을 수행한다. MAI-Thinking-1은 self-distillation checkpoint에서 RL climb을 재개하며, distillation recipe에 따라 이후 exploration 여지가 달라질 수 있다고 보고한다.

이 흐름은 자연스럽게 한 가지 질문으로 이어진다.

> **좋은 distillation endpoint는 곧 좋은 downstream RL initialization인가?**

우리는 distillation 직후의 benchmark score보다 distillation이 downstream RL에 어떤 exploration state를 넘기는가에 관심이 있다. 특히 token entropy가 낮아지는 것과 실제 reasoning path가 좁아지는 것을 구분한다. 모델은 대부분의 token에서는 매우 확신하면서도, 풀이 방향이 갈리는 몇몇 중요한 지점에서는 여러 reasoning branch를 유지할 수 있기 때문이다.

이를 세 가지 관측량으로 분석한다.

- \(H\): token-level uncertainty
- \(E\): semantic branch distribution
- \(V\): conditional branch success

\(H\)는 token 수준에서 policy가 얼마나 불확실한지를 나타낸다. \(E^{\mathrm{gen}}\)은 실제 생성이 여러 reasoning branch에 어떻게 분포하는지, \(E^{\mathrm{entry}}\)는 Base에서 검증된 branch 입구에 policy가 얼마나 많은 확률을 남겨 두었는지 나타낸다. \(V\)는 특정 branch를 선택한 rollout이 최종적으로 정답에 도달하는 비율이다.

Checkpoint 간 비교에는 이들로부터 두 개의 요약지표를 만든다. **Effective Branch Breadth**는 \(E^{\mathrm{gen}}\)을 이용해 모델이 실질적으로 몇 개의 reasoning route를 탐색하는지 나타낸다. **Budgeted Successful Branch Discovery**는 \(E^{\mathrm{gen}}\)과 \(V\)를 결합해, 주어진 RL rollout budget 안에서 몇 종류의 서로 다른 성공 경로를 발견할 수 있는지를 나타낸다.

이 측정체계를 이용해 Base → Distillation → RL 전 과정을 같은 branch space에서 추적하고, 추가 RL compute나 temperature 조절 같은 downstream intervention이 무엇을 얼마나 복구하는지도 비교한다.

만약 vanilla distillation이 student가 원래 접근할 수 있던 유효한 reasoning branch의 probability mass를 지나치게 깎는다는 결과가 나오면, 이를 막기 위한 **relative-floor projection**을 적용한다. Teacher를 최대한 따라가되, 각 token의 확률을 distillation 이전 student가 주던 값의 일정 비율 아래로는 떨어뜨리지 않는 방식이다. Floor가 절대 확률이 아니라 anchor 확률에 대한 비율이므로, 보호의 크기는 token마다 anchor의 판단을 따라간다.

이 연구의 목표는 새로운 distillation loss 하나를 제안하는 데 있지 않다. 궁극적으로 묻는 것은 다음이다.

> **Distillation이 만든 손상 가운데 무엇은 downstream에서 쉽게 복구되고, 무엇은 다음 stage로 넘기기 전에 보존해야 하는가?**

---

# 1. 문제 설정: 현재 성능과 다음 학습 가능성은 같은가?

Distillation과 RL은 모두 모델 성능을 높이지만 사용하는 학습 신호는 다르다. Distillation은 강한 teacher가 이미 알고 있는 행동을 student에게 전달한다. 반면 RLVR은 현재 student가 직접 생성한 rollout 가운데 verifier를 통과한 행동을 강화한다.

따라서 downstream RL의 출발점에서는 단순히 "무엇을 알고 있는가"뿐 아니라 어떤 행동을 스스로 생성할 수 있는가가 중요하다.

예를 들어 distillation 전 student가 서로 다른 세 가지 풀이 경로를 사용한다고 하자. Teacher가 그중 하나를 강하게 선호한다면 distillation 직후 pass@1은 올라갈 수 있다. 하지만 나머지 두 경로의 생성 확률이 거의 사라지면, downstream RL은 그 경로에서 성공 trajectory를 발견할 기회부터 잃는다. 현재 성능은 좋아졌지만 다음 학습 단계가 활용할 수 있는 exploration space는 오히려 좁아진 셈이다.

최근 SFT→RL 연구는 이미 두 목표가 일치하지 않을 수 있다는 징후를 보여준다. Quagmires는 높은 SFT 성능이 높은 post-RL 성능을 안정적으로 예측하지 못하며, Pass@large-\(k\)와 held-out generalization loss가 더 좋은 predictor가 될 수 있음을 보였다. AESL 역시 peak SFT checkpoint가 항상 가장 좋은 RL initialization은 아니며, SFT 중의 distributional forgetting과 diversity loss가 이후 RL에 영향을 줄 수 있음을 보였다.

우리는 여기서 한 단계 더 들어간다.

> **왜 immediate score가 비슷한 checkpoint들이 downstream RL에서는 다르게 성장하는가?**

그 차이를 token entropy 하나가 아니라 reasoning branch의 분포 변화에서 찾는다.

---

# 2. 왜 지금 중요한가: OPD가 training stage 사이의 연결 장치가 되고 있다

## 2.1 OPD는 이미 capability consolidation에 쓰이고 있다

최근 frontier technical report에서 OPD는 반복해서 등장한다.

DeepSeek-V4는 여러 domain specialist를 각각 SFT와 RL로 학습한 뒤 multi-teacher OPD로 하나의 unified policy에 통합한다. Kimi K3도 여러 전문 RL policy를 하나의 모델로 합치는 데 multi-teacher distillation을 사용한다. GLM-5는 reasoning, agentic, general capability를 순차적으로 학습하는 과정에서 앞 stage의 능력이 손상되는 문제를 cross-stage distillation으로 다룬다.

세 시스템의 세부 pipeline은 다르지만 공통점은 분명하다.

> **Distillation은 더 이상 마지막 compression 단계에만 머물지 않는다. 비싼 RL로 만든 capability를 다른 policy state로 옮기고 통합하는 핵심 연산자로 쓰이고 있다.**

이 사례들은 본 연구의 novelty를 뒷받침하기 위한 근거라기보다, OPD handoff가 실제 frontier-scale training에서 중요한 문제가 되었음을 보여주는 산업적 배경으로 사용한다.

## 2.2 Distillation checkpoint가 다시 학습의 출발점이 되는 사례도 있다

우리 문제와 더 직접적으로 맞닿는 사례도 있다.

Nemotron-Cascade 2는 multi-domain OPD를 Cascade RL의 중간 단계에 넣는다. 앞선 RL stage에서 얻은 능력을 distill해 regression을 복구한 뒤 다시 다음 RL stage로 넘어간다.

Qwen3.5-Omni도 specialist distillation과 OPD를 거친 뒤 interaction-aligned RL을 수행한다. 세부 목적은 본 연구의 수학 RLVR과 다르지만, OPD checkpoint를 downstream RL의 initialization으로 사용한다는 점은 같다.

MAI-Thinking-1은 OPD가 아니라 self-distillation을 사용하지만, distillation→RL handoff의 중요성을 가장 직접적으로 보여주는 사례다. RL-generated trace를 distill한 checkpoint에서 다시 RL climb을 시작하며, 어떤 trace를 얼마나 distill하느냐에 따라 이후 exploration 여지가 달라질 수 있다고 보고한다.

따라서 OPD→RLVR이라는 실험 설정은 현실에 없는 pipeline을 억지로 만든 것이 아니다. 이미 frontier training에서도 distillation된 policy가 다음 학습 stage의 출발점으로 쓰이고 있다. 다만 실제 pipeline에서는 domain shift, modality shift, objective shift가 함께 일어난다. 우리는 같은 reasoning domain과 같은 verifier를 유지해 이러한 confound를 걷어내고, distillation 자체가 downstream RL initialization에 미치는 영향만 통제해서 측정한다.

---

# 3. 핵심 분석: distillation이 RL에 넘기는 exploration state

핵심은 distillation이 모델을 단순히 더 confident하게 만드는 것인지, 아니면 downstream RL이 활용할 수 있는 reasoning path 자체를 좁히는 것인지 구분하는 데 있다. 이를 위해 token 수준에서는 \(H\), branch 수준에서는 \(E\)와 \(V\)를 측정한다. 모든 비교는 학습 데이터와 분리된 고정 branch panel 위에서 수행하며, 같은 분석 공간에서 Base → Distillation → RL의 변화를 추적한다.

## 3.1 고정 reasoning branch panel

Branch는 문체나 opening token이 아니라, **한 문제를 푸는 서로 구별되는 수학적 접근법**이다. 표현이 다르더라도 핵심 논리가 같으면 같은 branch로 묶고, 하나의 응답에 여러 아이디어가 섞여 있더라도 judge가 판정한 primary approach 하나에만 배정한다. 애매하거나 유효하지 않은 응답은 억지로 기존 branch에 넣지 않고 `other`, `ambiguous`, `failed`로 남긴다.

Panel 후보는 두 경로에서 모은다. 외부 API 모델은 서로 다른 계열의 모델 여섯 개에서 문제당 완결 풀이 16개씩 생성해, Base 하나만 보아서는 놓칠 수 있는 접근법을 넓게 찾는다. Base는 정답 완결문을 직접 모으는 대신 자신의 policy 내부를 재귀적으로 탐색한다. Opening의 첫 두 token에서는 확률 1% 이상인 후보를 전개하고, 그 뒤에는 greedy로 진행하다가 entropy가 1.5 이상이면서 확률 1% 이상 후보가 둘 이상인 지점에서 다시 갈라진다. 이 과정을 depth 3까지 반복하며 다음 fork를 찾는 길이는 단계마다 최대 48 token이다.

Base tree의 leaf는 아직 풀이도, branch도 아니다. Judge가 partial prefix에서 보이는 primary approach를 먼저 판정하고, 같은 prefix에서 Qwen3-14B를 네 번 완결시킨다. 기존 verifier를 통과하면서 prefix의 접근법을 유지한 완결이 한 번이라도 나오면 그 leaf를 viable entry로 받아들인다. 이 검증은 Base가 직접 정답을 완성할 수 있다는 뜻이 아니라, 그 prefix가 유효한 풀이 경로의 입구인지 확인하는 최소 타당성 검사다.

검증을 통과한 Base leaf와 정답인 API 완결문을 한데 모은 뒤, judge가 문제별 접근법 목록을 만든다. 같은 접근법은 생성 모델이나 표현이 달라도 하나로 병합한다. 이때 Base에서 검증된 entry가 있는 branch는 Base가 원래 접근 가능한 경로로 표시하고, API에서만 발견된 branch는 panel coverage에는 포함하되 “distillation이 Base에서 잃게 한 경로”라는 주장에는 사용하지 않는다.

원문 embedding은 branch를 정의하거나 응답을 배정하는 도구로 쓰지 않는다. 사전 실험에서 의미보다 생성 모델의 문체와 출처가 거리를 더 강하게 좌우했기 때문이다. Embedding은 judge가 만든 semantic branch를 고정한 뒤, checkpoint별 응답 분포를 문제별 small-multiple로 보여주는 보조 시각화에만 사용한다.

한 번 만든 panel과 판정 기준은 Base, Distill, EMBER, RL의 모든 checkpoint에 동일하게 적용한다. 구축 절차와 고정값, Base/API branch의 관계는 [Branch Panel and E](Branch_Panel_and_E.md)에 정리한다.

## 3.2 Token-level uncertainty \(H\)

Branch-level exploration과 비교하기 위한 기준선으로 token entropy를 측정한다. 각 reasoning position에서

\[
H_t
=
-\sum_v p(v\mid s_t)\log p(v\mid s_t)
\]

를 계산하고 response 전체의 평균과 위치별 분포를 기록한다.

높은 entropy 자체가 목표는 아니다. \(H\)는 policy가 token 수준에서 얼마나 stochastic한지를 보여주는 익숙한 기준선이다. 우리가 확인하려는 것은 token entropy가 비슷하거나 함께 낮아지더라도 실제 reasoning path의 폭은 다를 수 있는가이다. 대부분의 token에서는 매우 confident하면서도, 풀이 방향이 갈리는 몇몇 지점에서는 여러 reasoning alternative를 유지할 수 있기 때문이다.

## 3.3 Semantic branch distribution \(E\)

고정 panel 위의 분포는 생성 기반 \(E^{\mathrm{gen}}\)과 prefill 기반 \(E^{\mathrm{entry}}\)로 나누어 본다. 두 값은 서로를 대신하지 않는다. 전자는 모델이 실제 생성에서 어떤 접근법을 얼마나 자주 사용하는지 보여주고, 후자는 각 접근법의 입구에 얼마만큼의 확률을 남겨 두었는지 보여준다.

**생성 기반 측정.** 각 checkpoint에서 동일한 sampler로 문제당 \(N\)개의 완결 응답을 생성하고, judge가 각 응답을 고정 panel의 primary approach 하나에 hard assignment한다.

\[
E^{\mathrm{gen}}_{\theta,j}
=
\frac{n_{\theta,j}}{N}
\]

분모에는 성공 응답만이 아니라 모든 생성이 들어간다. `other`, `ambiguous`, `failed`도 별도 칸으로 남기므로, 낮은 정답률이나 panel 밖 응답이 기존 branch의 질량으로 잘못 정규화되지 않는다. 이 분포가 checkpoint별 실제 행동 범위를 나타내는 주 측정이다.

**Prefill 기반 측정.** Base에서 검증된 branch \(j\)마다 entry prefix 집합 \(P_j\)를 고정한다. Checkpoint가 각 prefix에 부여하는 teacher-forced probability를 계산해 합하면

\[
E^{\mathrm{entry}}_{\theta,j}
=
\sum_{p\in P_j}P_\theta(p\mid x)
\]

를 얻는다. 이 값은 branch 전체의 정확한 확률이 아니라, panel이 확보한 **고정 입구들의 covered mass**다. 같은 접근법을 다른 표현으로 시작하는 모든 경우를 열거할 수 없기 때문이다. 따라서 entry mass의 하락만으로 branch가 완전히 사라졌다고 단정하지 않고, 생성 기반 분포와 함께 해석한다. API-only branch의 대표 entry 구성은 실제 panel을 본 뒤 결정한다.

예를 들어 Base의 생성 분포가 \((0.45,0.35,0.20)\)인데 Distill 이후 \((0.90,0.08,0.02)\)가 되었다면 실제 생성이 한 접근법으로 집중된 것이다. 동시에 나머지 branch의 entry mass도 크게 줄었다면, 단지 표본에서 덜 나온 것을 넘어 해당 경로의 입구 자체가 약해졌다는 근거가 된다. 반대로 entry mass는 유지되는데 생성 빈도만 줄었다면 접근 가능성은 남아 있으나 자연 생성에서 선택되지 않는 상태로 해석할 수 있다.

### Soft entry loss와 hard entry loss

모든 생성 빈도 감소가 같은 의미를 갖는 것은 아니다. **Soft entry loss**는 branch의 발생 빈도가 크게 줄었지만 주어진 RL rollout budget 안에서는 여전히 일정 확률로 발견되는 경우다. 반면 **hard entry loss**는 생성 확률이 너무 작아 실제 rollout budget 안에서는 사실상 관찰되지 않는 경우다.

이 구분은 임의의 raw-probability threshold가 아니라 실제 downstream budget을 기준으로 한다. Branch \(j\)의 생성 확률이 \(E^{\mathrm{gen}}_j\)이고 한 문제에서 \(G\)번 sampling할 기회가 있다면, 적어도 한 번 해당 branch를 볼 확률은

\[
P_{\mathrm{hit},j}(G)
=
1-(1-E^{\mathrm{gen}}_j)^G
\]

이다. Soft/hard를 나누는 구체적인 hit-probability 기준은 panel 구축 뒤 실측 분포를 보고 고정한다. 이후 recovery 분석에서는 temperature 조절이나 reheating이 soft loss는 복구하면서 hard loss에는 거의 영향을 주지 않는지 확인한다.

## 3.4 Conditional branch success \(V\)

Branch가 다양하다는 사실만으로 그 exploration이 RL에 유용하다고 말할 수는 없다. 따라서 같은 자연 rollout에서 branch별 조건부 성공률도 함께 측정한다.

Branch \(j\)로 들어간 rollout 가운데 최종 verifier를 통과한 비율을

\[
V_{\theta,j}(s)
=
P_\theta(R=1\mid C_{s,j},s)
\]

로 정의한다. 예를 들어 \(C_2\)로 들어간 rollout이 50개이고 이 가운데 15개가 정답이라면 \(V_{\theta,2}=0.30\)이다.

\(V\)는 branch 자체의 보편적인 품질을 뜻하지 않는다. 현재 모델이 해당 reasoning route를 자연스럽게 선택했을 때 얼마나 성공하는가를 나타내는 조건부 성능이다. \(E\)와 \(V\)를 함께 보면 branch selection의 변화와 branch 내부 execution의 변화를 구분할 수 있다.

Panel 구축 때 Qwen3-14B로 수행하는 K=4 leaf validation과도 구분해야 한다. 전자는 Base prefix가 유효한 풀이로 이어질 가능성이 한 번이라도 있는지를 확인하는 일회성 수용 검사이고, \(V_{\theta,j}\)는 각 checkpoint가 자연 생성에서 그 branch를 선택했을 때 실제로 성공하는 비율이다.

\[
E^{\mathrm{gen}}\downarrow,\qquad V\approx\text{constant}
\]

라면 branch의 발생 빈도는 줄었지만 자연스럽게 그 branch가 나왔을 때의 성공률은 크게 변하지 않은 것이다. 반대로

\[
E^{\mathrm{gen}}\approx\text{constant},\qquad V\downarrow
\]

라면 branch 자체는 계속 선택되지만 해당 route에서의 성공 가능성이 낮아진 것이다.

### Forced-prefix probe: 선택적 causal check

자연 rollout에서 측정한 \(V\)는 observational statistic이다. 특히 \(E^{\mathrm{gen}}_j\)가 매우 작은 branch에서는 표본 수가 부족할 수 있다. 따라서 중요한 희귀 Base-validated branch에 한해서는 해당 branch의 짧은 entry prefix를 강제로 넣은 뒤 continuation을 반복 생성하는 별도 probe를 사용한다.

이 실험은 \(V\)의 기본 정의를 바꾸기 위한 것이 아니다. 자연 rollout에서는 branch가 거의 사라졌더라도, 그 reasoning route를 실행할 capability 자체는 남아 있는가를 확인하기 위한 causal check다. 따라서 "distillation이 capability를 지운 것이 아니라 access를 지웠다"와 같은 강한 주장은 자연 \(E,V\) 통계와 forced-prefix 결과가 같은 방향을 보일 때만 사용한다.

## 3.5 모델 수준의 reasoning exploration

Branch별 \(E^{\mathrm{gen}}_j\), \(E^{\mathrm{entry}}_j\), \(V_j\)는 변화의 원인을 분석하는 데 적합하지만 여러 checkpoint를 한눈에 비교하기에는 복잡하다. 따라서 문제별 생성 분포에서 두 개의 요약지표를 계산한다.

### Effective Branch Breadth

정상 semantic branch에 배정된 질량을 \(m_{\mathrm{asg}}=\sum_jE^{\mathrm{gen}}_j\)라고 두고

\[
\widetilde E^{\mathrm{gen}}_j
=
\frac{E^{\mathrm{gen}}_j}{m_{\mathrm{asg}}}
\]

로 정규화한다. 이 분포가 실질적으로 몇 개의 reasoning route에 걸쳐 있는지를

\[
B(s)
=
\exp\left(
-\sum_j\widetilde E^{\mathrm{gen}}_j(s)\log \widetilde E^{\mathrm{gen}}_j(s)
\right)
\]

로 계산한다. 이를 **Effective Branch Breadth**라고 부른다.

한 branch에 거의 모든 probability가 몰려 있으면 \(B\)는 1에 가까워지고, 여러 branch를 비슷한 빈도로 사용하면 값이 커진다. 여러 held-out 문제를 비교할 때는 \(B(s)\)를 문제별로 계산한 뒤

\[
\overline B
=
\frac{1}{N}\sum_{i=1}^{N}B(s_i)
\]

를 사용한다. `other`, `ambiguous`, `failed`는 breadth 계산에서 제외하되 그 질량을 함께 보고해, 실패가 늘어난 모델을 넓거나 좁은 모델로 잘못 읽지 않는다.

### Budgeted Successful Branch Discovery

Reasoning branch가 많더라도 모두 실패한다면 RL에는 큰 도움이 되지 않는다. Branch \(j\)에서 성공 trajectory가 나올 확률은

\[
q_j
=
E^{\mathrm{gen}}_jV_j
=
P(C_j,R=1)
\]

이다. 한 prompt에서 downstream RL이 \(G\)개의 rollout을 생성한다면, branch \(j\)에서 성공 trajectory를 적어도 하나 관찰할 확률은

\[
1-(1-q_j)^G
\]

가 된다. 따라서

\[
D_{\mathrm{succ}}(s;G)
=
\sum_j\left[1-(1-q_j)^G\right]
\]

를 **Budgeted Successful Branch Discovery**라고 정의한다.

이 값은 주어진 RL rollout budget 안에서 평균적으로 몇 종류의 서로 다른 성공 reasoning route를 발견할 수 있는가를 나타낸다. Pass@\(G\)가 성공 trajectory가 하나라도 존재하는지만 본다면, \(D_{\mathrm{succ}}(G)\)는 그 성공 가능성이 몇 개의 reasoning mode에 걸쳐 있는지까지 반영한다.

모델 수준에서는

\[
\overline D_{\mathrm{succ}}(G)
=
\frac{1}{N}\sum_{i=1}^{N}D_{\mathrm{succ}}(s_i;G)
\]

를 사용한다. `other`나 `ambiguous` 질량이 큰 checkpoint에서는 \(D_{\mathrm{succ}}\)를 고정 panel 안에서만 계산된 보수적인 추정으로 해석한다.

## 3.6 Base → Distill → RL의 궤적과 RL 학습 신호

모든 checkpoint는 한 번 동결한 같은 semantic branch panel에서 평가한다. 기본 궤적은

\[
\text{Base (= frozen anchor)}
\rightarrow
\text{Vanilla Distillation}
\rightarrow
\text{RL}
\]

이고, 제안 방법은

\[
\text{Anchor}
\rightarrow
\text{Protected Distillation}
\rightarrow
\text{RL}
\]

로 비교한다. RL 중간 checkpoint도 저장해 final model만 보는 것이 아니라 exploration 구조가 언제, 어떻게 바뀌는지를 추적한다.

Vanilla와 Protected Distillation은 **matched-distillation protocol**로 비교한다. Distillation 직후 held-out pass@1을 가능한 한 맞추고, output length와 기본 generation statistics도 함께 확인한다. Teacher, student initialization, distillation compute, optimizer, rollout 조건은 동일하게 유지한다. 그래야 이후 RL의 차이를 immediate performance가 아니라 branch preservation의 차이로 해석할 수 있다.

모델 수준의 대표 비교에는 **Token Entropy \(H\)**, **Effective Branch Breadth \(\overline B\)**, **Budgeted Successful Branch Discovery \(\overline D_{\mathrm{succ}}(G)\)**를 사용한다. Branch별 \(E^{\mathrm{gen}}_j\), \(E^{\mathrm{entry}}_j\), \(V_j\)는 이 차이가 어디에서 생겼는지 설명하는 mechanistic analysis에 사용한다.

Base, Distillation, RL이 반드시 정해진 방향으로 움직인다고 가정하지는 않는다. 우리가 확인하려는 것은 비슷한 immediate performance와 낮은 token entropy를 가진 모델 사이에서도 branch breadth와 successful discovery가 크게 달라질 수 있는지, 그리고 그 차이가 fixed-budget RL 결과로 이어지는지다.

### 기존 predictor 대비 추가 설명력

Branch-level 지표가 유용하려면 기존의 강한 pre-RL predictor가 이미 설명하는 내용을 다시 재는 데 그쳐서는 안 된다. Quagmires가 제시한 **held-out loss**와 **Pass@large-\(k\)**를 기준 predictor로 두고, 각 distillation recipe와 seed에서 이 값들과 \(H\), \(\overline B\), \(\overline D_{\mathrm{succ}}(G)\)를 함께 기록한다.

여기서 주된 질문은 \(\overline D_{\mathrm{succ}}(G)\)가 Pass@large-\(k\)를 단순히 대체하느냐가 아니다. held-out loss와 Pass@large-\(k\)를 이미 알고 있는 상태에서도 branch-level 정보가 post-RL outcome을 추가로 설명하는가를 본다. 따라서 baseline predictor만 사용한 경우와 여기에 \(\overline B\) 또는 \(\overline D_{\mathrm{succ}}(G)\)를 추가한 경우의 held-out prediction과 rank correlation을 비교한다. 특히 Pass@large-\(k\)와 immediate performance가 비슷하지만 \(\overline D_{\mathrm{succ}}(G)\)가 다른 checkpoint pair가 이후 RL에서도 같은 순서로 갈리는지를 중요하게 본다.

이 분석에서 독립 표본의 기본 단위는 distillation recipe × seed run으로 둔다. 같은 run의 intermediate checkpoint 여러 개를 서로 독립된 모델처럼 세어 predictor 성능을 부풀리지 않는다. \(\overline B\)는 주로 exploration 구조를 설명하는 mechanistic summary로, \(\overline D_{\mathrm{succ}}(G)\)는 fixed rollout budget과 직접 대응하는 predictor candidate로 다룬다.

### Token-level floor가 branch access로 이어지는지 검증

제안 방법은 token probability에 직접 개입하지만, 우리가 중요하게 보는 결과는 branch-level \(E^{\mathrm{entry}}_j\)와 \(E^{\mathrm{gen}}_j\)다. 따라서 token-level protection이 실제 branch entry와 생성 빈도의 회복으로 이어지는지 별도로 검증한다.

Base-validated branch \(C_{s,j}\)마다 panel 구축 단계에서 고정한 entry prefix를 사용한다. 두 arm에 같은 prefix 집합을 teacher forcing으로 넣어 \(E^{\mathrm{entry}}_{\theta,j}\)를 계산하고, 해당 구간에서 relative floor가 얼마나 자주 bind하며 teacher target 대비 probability를 얼마나 끌어올리는지 기록한다. 다음으로 protected student가 vanilla student보다 같은 branch의 covered entry mass를 더 많이 유지하는지 확인한다. 마지막으로 canonical sampler에서 실제 생성 빈도 \(E^{\mathrm{gen}}_{\theta,j}\)가 함께 회복되는지를 측정한다.

즉 branch별로 다음 세 단계가 같은 방향으로 정렬되는지를 본다.

\[
\text{floor binding / target lift}
\rightarrow
\Delta E^{\mathrm{entry}}_j
\rightarrow
\Delta E^{\mathrm{gen}}_j
\]

Floor가 거의 bind하지 않은 Base-validated branch는 자연스러운 negative control이 된다. 이 branch들에서는 entry mass와 생성 빈도도 크게 변하지 않아야 한다. 반대로 floor가 강하게 작동한 branch일수록 두 값이 함께 회복된다면, relative-floor가 단순한 token-level regularizer가 아니라 실제 reasoning-mode accessibility를 복원한다는 근거가 된다.

### RL 학습 신호와의 연결

Branch-level exploration이 실제 RL optimization으로 이어지는 과정도 함께 기록한다. GRPO group에서 모든 rollout이 실패한 **all-fail group**, 모든 rollout이 성공한 **all-success group**, 성공과 실패가 함께 존재하는 **informative group**의 비율을 측정한다. 전체적으로 검증하려는 mechanistic chain은 다음과 같다.

\[
\text{Base-validated branch entry에서 floor가 작동}
\rightarrow
E^{\mathrm{entry}}_j\text{ 증가}
\rightarrow
E^{\mathrm{gen}}_j\text{ 회복}
\rightarrow
\overline D_{\mathrm{succ}}(G)\text{ 증가}
\rightarrow
\text{all-fail 감소}
\rightarrow
\text{informative group 증가}
\rightarrow
\text{더 나은 fixed-budget RL}
\]

All-fail이나 informative-group 비율은 새로운 headline metric이 아니다. 이 값들은 \(\overline D_{\mathrm{succ}}(G)\)와 실제 RL learning signal 사이의 연결을 확인하는 중간 진단으로 사용한다.

이 연결이 데이터에서 확인된다면 핵심 주장은 단순히 "다양성이 높을수록 좋다"가 아니다.

> **Distillation이 어떤 reasoning mode를 RL sampler의 가시권 밖으로 밀어내는지가 이후 학습 가능성에 영향을 주며, token entropy만으로는 이 손상을 충분히 설명할 수 없다.**

---

# 4. Recovery analysis: downstream에서 무엇까지 복구되는가?

Distillation 직후 checkpoint에 downstream intervention을 적용해 exploration 손상이 실제로 얼마나 돌아오는지 측정한다. 기본 intervention은 두 가지다. 추가 RL compute는 기존 RL run을 연장해 같은 checkpoint에 compute를 더 주는 것이고, rollout temperature 조절은 추가 학습 없이 sampling 설정만 바꾸는 것이다. 더 비싼 intervention(entropy-controlled RL, self-reheating 또는 self-distillation, pre-distillation anchor를 다시 사용하는 repair)은 이 두 가지로 복구되지 않는 손상이 확인될 때만 확장한다.

각 intervention이 끝난 뒤에는 sampling 설정을 다시 동일한 canonical sampler로 되돌린다. 모델 수준에서는 \(H\), \(\overline B\), \(\overline D_{\mathrm{succ}}(G)\)를 중심으로 비교하고, 필요한 경우 branch별 \(E^{\mathrm{gen}}_j\), \(E^{\mathrm{entry}}_j\), \(V_j\)를 이용해 변화의 원인을 분해한다. 이렇게 해야 "temperature를 높여둔 동안만 다양해진 것"과 "policy 자체의 exploration structure가 실제로 회복된 것"을 구분할 수 있다.

Recovery는 두 조건을 함께 본다. 첫째는 compute다. 같은 총 post-handoff compute에서 얼마나 복구되는지를 비교한다. 둘째는 information access다. 현재 damaged checkpoint와 reward만 사용하는지, 자기 logits가 필요한지, pre-distillation anchor나 teacher information을 다시 사용해야 하는지를 구분한다.

따라서 어떤 손상이 절대적으로 irrecoverable하다고 주장하지 않는다. 대신 다음을 묻는다.

> **같은 budget과 information access에서 어떤 손상은 쉽게 돌아오고, 어떤 손상은 그렇지 않은가?**

특히 \(H\)는 회복됐는데 \(\overline B\)는 돌아오지 않는지, \(\overline B\)는 회복됐는데 \(\overline D_{\mathrm{succ}}\)는 돌아오지 않는지, 그리고 단순히 RL compute를 더 주는 것보다 과거 anchor information을 다시 쓰는 편이 더 효과적인지를 비교한다.

---

# 5. 분석에서 도출되는 개선 방법

## 5.1 Relative-floor projection

분석에서 vanilla distillation이 Base-validated branch의 \(E^{\mathrm{entry}}_j\)와 \(E^{\mathrm{gen}}_j\)를 과도하게 낮추고, 그 결과 \(\overline B\)와 \(\overline D_{\mathrm{succ}}\)가 떨어진다는 현상이 확인되면 entry preservation을 적용한다.

Frozen pre-distillation anchor를 \(\pi_A\), teacher를 \(\pi_T\)라고 하자. State \(s\)에서 target distribution을

\[
q^*
=
\arg\min_q D_{\mathrm{KL}}(q\|\pi_T)
\]

subject to

\[
q(v)\ge\beta\,\pi_A(v),
\qquad \forall v
\]

로 정의한다. Floor는 절대 확률이 아니라 anchor 확률에 대한 비율이므로 anchor가 이미 낮은 확률만 주던 token에서는 floor도 그만큼 낮아 실질적인 제약이 되지 않는다. 보호의 크기가 token마다 anchor의 판단을 따라가는 셈이다. 또한 \(\beta<1\)이면 floor의 총합이 \(\beta\)로 1보다 작으므로 제약은 항상 실현 가능하다.

직관은 간단하다.

> **Teacher는 최대한 그대로 따라가되, distillation 이전 student가 실제로 사용하던 candidate를 일정 수준 아래로 떨어뜨리지 않는다.**

이 문제의 해는 닫힌 형태로 주어진다.

\[
q^*(v)
=
\max\left(c\,\pi_T(v),\ \beta\,\pi_A(v)\right)
\]

여기서 \(c\)는 전체 합이 1이 되도록 정해지는 공통 정규화 상수다. 제약이 걸리지 않은 token에서는 teacher의 비율을 유지하는 것이 최적이고, 제약이 걸린 token은 정확히 floor까지만 올리는 것이 KL을 최소화하기 때문이다. 따라서 floor를 위반한 candidate만 floor에 고정되고, 나머지 token은 공통 상수 \(c\)로 축소되어 clamp되지 않은 token 사이에서는 teacher의 상대 선호가 그대로 보존된다. Target은 state마다 내부 최적화 루프 없이 floor 계산, 위반 candidate clamp, 나머지 재정규화의 한 번의 pass로 얻어지며 비용은 softmax 자체와 같은 차수다.

Teacher와 anchor를 전역적으로 섞는 arithmetic mixture와 달리, 실제 floor를 위반한 candidate만 수정한다. 이 token-level 개입이 실제 branch access를 복구하는지는 §3.6에서 floor binding → \(E^{\mathrm{entry}}_j\) 증가 → \(E^{\mathrm{gen}}_j\) 회복의 순서로 직접 확인한다.

## 5.2 \(V\)가 독립적으로 무너지는 경우

주 방법은 우선 branch entry preservation에 집중한다. 비슷한 \(E^{\mathrm{gen}}\)을 가진 branch들 사이에서도 distillation 이후 \(V\)가 반복적으로 크게 낮아지고, 이 변화가 post-RL 성능을 추가로 설명한다면 그때만 short-horizon continuation preservation을 두 번째 방법으로 확장한다.

반대로 \(V\)가 대체로 안정적이라면 continuation method는 제거한다. 이 경우 더 단순하면서도 강한 결론을 얻을 수 있다.

> **Distillation은 reasoning capability 자체를 지우기보다, 그 capability에 접근하는 probability를 주로 깎는다.**

---

# 6. Prevention과 repair 비교

최종 평가는 benchmark improvement만으로 끝내지 않는다. 같은 branch-access 손상을 **prevention**과 **repair** 두 방식으로 다룬다.

Prevention에서는 distillation 중 relative-floor를 적용해 branch access를 처음부터 보존한다. Repair에서는 vanilla distillation 이후 downstream intervention으로 branch access를 다시 복구한다. 두 방식은 가능한 한 같은 total compute와 같은 information access에서 비교한다.

만약 prevention이 같은 \(\overline B\) 또는 \(\overline D_{\mathrm{succ}}(G)\)에 더 적은 비용으로 도달한다면,

> **이 exploration structure는 downstream에서 수리하기보다 handoff 전에 보존하는 편이 효율적이다.**

라는 pipeline-level 원리를 얻는다. 반대로 downstream repair가 싸고 충분히 잘 작동한다면 distillation objective를 복잡하게 만들 이유가 없다는 반대 결론도 가능하다.

---

# 7. 실험 계획

## 7.1 모델과 training pipeline

주 student는 Qwen3-1.7B-Base를 사용한다. Base 자체를 frozen pre-distillation anchor로 사용하며, 별도의 cold-start 단계는 두지 않는다. 이후 같은 tokenizer 계열의 강한 Qwen teacher로 OPD를 수행하고, 동일한 fixed-budget RLVR을 적용한다.

핵심 비교는

\[
\text{Anchor}
\rightarrow
\text{Vanilla OPD}
\rightarrow
\text{RL}
\]

과

\[
\text{Anchor}
\rightarrow
\text{Protected OPD}
\rightarrow
\text{RL}
\]

이다. 두 distillation 조건은 §3.6의 matched-distillation protocol에 따라 immediate pass@1과 기본 generation statistics를 가능한 한 맞춘다. RL 중간 checkpoint도 저장해 branch state가 언제 바뀌는지 추적한다.

Main 결과가 확인되면 더 큰 Qwen student에서 축소 transfer를 수행한다.
