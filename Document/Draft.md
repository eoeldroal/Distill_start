# Distillation → RL 인계의 탐색 구조와 회복 가능성

## 요약

최근 frontier model의 post-training은 SFT, RL, distillation을 한 번씩 거치는 선형 pipeline이 아니다. 여러 training stage를 오가며 capability를 만들고, 합치고, 다시 강화하는 과정에 가깝다. 이 과정에서 on-policy distillation(OPD)은 단순한 model compression을 넘어, 서로 다른 RL specialist의 능력을 하나의 policy로 통합하거나 stage 사이에서 capability를 옮기는 수단으로 쓰이기 시작했다.

DeepSeek-V4, Kimi K3, GLM-5는 OPD 또는 cross-stage distillation을 대규모 capability consolidation에 활용한다. Nemotron-Cascade 2는 OPD를 RL stage 사이의 중간 단계로 배치하고, Qwen3.5-Omni도 OPD 이후 다시 RL을 수행한다. MAI-Thinking-1은 self-distillation checkpoint에서 RL climb을 재개하며, distillation recipe에 따라 이후 exploration 여지가 달라질 수 있다고 보고한다.

이 흐름은 자연스럽게 한 가지 질문으로 이어진다.

> **좋은 distillation endpoint는 곧 좋은 downstream RL initialization인가?**

우리는 distillation 직후의 benchmark score보다 distillation이 downstream RL에 어떤 exploration state를 넘기는가에 관심이 있다. 특히 token entropy가 낮아지는 것과 실제 reasoning path가 좁아지는 것을 구분한다. 모델은 대부분의 token에서는 매우 확신하면서도, 풀이 방향이 갈리는 몇몇 중요한 지점에서는 여러 reasoning branch를 유지할 수 있기 때문이다.

이를 세 가지 관측량으로 분석한다.

- \(H\): token-level uncertainty
- \(E\): branch entry distribution
- \(V\): conditional branch success

\(H\)는 token 수준에서 policy가 얼마나 불확실한지를 나타낸다. \(E\)는 실제 downstream sampler에서 probability mass가 여러 reasoning branch에 어떻게 배분되는지를 나타낸다. \(V\)는 특정 branch를 선택한 rollout이 최종적으로 정답에 도달하는 비율이다.

Checkpoint 간 비교에는 이들로부터 두 개의 요약지표를 만든다. **Effective Branch Breadth**는 \(E\)를 이용해 모델이 실질적으로 몇 개의 reasoning route를 탐색하는지 나타낸다. **Budgeted Successful Branch Discovery**는 \(E\)와 \(V\)를 결합해, 주어진 RL rollout budget 안에서 몇 종류의 서로 다른 성공 경로를 발견할 수 있는지를 나타낸다.

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

Branch 분석은 별도의 held-out diagnostic set에서 수행한다. 주 도메인은 수학으로 두고, reasoning 과정이 중요하면서 최종 답은 verifier로 명확하게 판정할 수 있는 competition-style problem을 사용한다. Diagnostic set은 측정 방식을 정하기 위한 `branch-dev`와 실제 checkpoint 비교에 사용하는 `branch-test`로 나눈다. Continuation 길이, fork 탐지 기준, embedding과 clustering 설정은 branch-dev에서 고정하고 branch-test에서는 변경하지 않는다.

### Opening state: 모델 수준 exploration의 기본 단위

가장 중요한 분석 단위는 문제를 받은 직후의 opening state다. 모든 모델과 rollout이 같은 state에서 출발하므로, 여기서의 branch distribution은 모델의 end-to-end exploration과 직접 연결된다. 따라서 모델 수준의 Effective Branch Breadth와 Budgeted Successful Branch Discovery는 opening state를 기준으로 보고한다.

### Internal fork: branch collapse의 위치를 보는 probe

Opening 이후 reasoning 내부의 분기점은 별도의 mechanistic probe로 사용한다. Internal fork candidate는 pre-distillation anchor의 자연 trajectory에서 찾는다. Anchor trajectory를 따라가며 next-token entropy가 주변보다 뚜렷하게 높아지는 위치를 후보로 잡고 해당 prefix를 고정한다. 이후 모든 checkpoint는 같은 state \(s\)를 조건으로 비교한다.

Entropy가 높다는 이유만으로 해당 위치를 fork라고 보지는 않는다. 같은 state에서 continuation을 반복 생성했을 때 서로 다른 안정적인 continuation mode가 실제로 나타날 때만 internal fork로 남긴다. Opening state가 모델 전체의 exploration을 보는 기본 단위라면, internal fork는 어디에서 branch collapse가 생기는지를 들여다보는 probe다.

### Branch discovery와 고정

각 candidate state \(s\)에서 32~64 token 길이의 짧은 continuation을 충분히 많이 생성한다. Discovery의 목적은 특정 checkpoint를 평가하는 것이 아니라 가능한 reasoning alternative를 폭넓게 찾는 데 있다. 따라서 frozen anchor 외에 teacher와 같은 계열의 강한 모델을 보조적으로 활용할 수 있고, sampler도 실제 RL보다 다양성을 넓히는 방향으로 설정할 수 있다.

여기서 중요한 것은 discovery와 measurement를 분리하는 것이다. 여러 모델과 exploratory sampler는 branch space를 만드는 데만 사용하고, 실제 checkpoint의 \(E\)를 측정할 때는 downstream RL과 동일한 canonical sampler를 사용한다.

Discovery에 여러 소스를 쓰면 continuation의 문체가 소스마다 달라진다. 이 이질성은 branch space의 폭을 넓히는 대가이므로 줄이지 않되, checkpoint 비교 결과는 소스별로 층화해 보고한다. 층화하면 문체 효과가 층 안에 갇히고, 결과가 특정 소스에서만 나온 것이 아님을 함께 보일 수 있다.

Canonical sampler는 truncation 없는 full softmax로 두고 temperature는 RL rollout과 동일하게 고정한다. 따라서 모델이 부여하는 확률과 sampler가 실제로 뽑는 확률이 일치하며, rollout 분포와 policy 분포가 어긋나지 않는다.

생성된 continuation

\[
r_1,\ldots,r_M
\]

을 동일한 representation으로 embedding한 뒤, 각 reasoning state 안에서만 clustering한다.

\[
\{r_1,\ldots,r_M\}
\rightarrow
\{C_{s,1},\ldots,C_{s,J}\}.
\]

Branch는 사람이 미리 이름 붙인 풀이 전략이 아니다. 특정 state에서 반복적으로 나타나는 안정적인 local continuation cluster 자체를 branch로 본다. 표현만 다른 continuation은 같은 cluster에 들어갈 수 있고, 어느 cluster에도 안정적으로 속하지 않는 sample은 `ambiguous/noise`로 남긴다.

Clustering 품질은 branch-dev에서 cluster separation, bootstrap stability, ambiguous 비율과 소규모 human audit으로 확인한다.

이 지표들은 cluster가 잘 갈라졌는지는 말해 주지만 무엇을 기준으로 갈라졌는지는 말해 주지 않는다. 문체로 깔끔하게 갈라진 cluster도 이 검사를 전부 통과한다. 따라서 embedding 공간이 문체가 아니라 풀이 방법을 반영하는지 확인하는 절차를 따로 둔다.

절차는 이렇다. 각 state의 continuation에 embedding과 무관한 경로로 approach label을 부여한 뒤, 쌍을 두 종류로 나눈다. 방법만 공유하는 쌍(같은 방법, 다른 소스)과 소스만 공유하는 쌍(다른 방법, 같은 소스)이다. 두 종류는 공통점을 하나씩만 갖고 그 하나가 서로 다르므로, 어느 쪽이 더 가까운지가 공간이 무엇을 공통점으로 치는지를 그대로 드러낸다. 소스만 공유한 쌍이 더 가깝다면 cluster는 branch가 아니라 문체 집단이고, 그 위에서 잰 \(E\) 변화는 방법의 변화로 읽을 수 없다. 이 검사의 실측과 그로부터 나온 측정 규칙은 Cal_E_Before_train.md에 있다.

LLM-as-a-judge는 핵심 branch assignment에는 사용하지 않는다. 다만 위 수용 검사의 approach label처럼, assignment가 아니라 embedding 공간을 검증하기 위한 독립 기준을 만드는 데는 사용한다. 이때 판정자는 텍스트를 쓴 모델을 알 수 없어야 하고, 문제별로 미리 고정한 label 목록에서만 고르며, 문체를 판단 근거로 삼지 않는다는 지침을 받는다.

한 번 만든 branch space는 이후 모든 checkpoint에 대해 고정한다. Base, Distill, RL마다 clustering을 다시 하면 같은 reasoning alternative의 probability가 어떻게 변했는지 비교할 수 없기 때문이다.

Discovery에 여러 모델을 사용하면 pre-distillation student가 원래 갖고 있지 않던 branch도 포함될 수 있다. 따라서 전체 branch universe와 별개로, frozen anchor에서 실제로 의미 있는 probability mass를 가졌던 **inherited branch subset**을 표시해 둔다. Base → Distill → RL의 전체 궤적은 모든 안정적 branch에서 관찰하되, "distillation이 무엇을 잃게 했는가"와 relative-floor의 작동 기제를 분석할 때는 inherited branch를 중심으로 본다. Inherited 여부를 정하는 최소 anchor mass 기준 \(m_{\mathrm{min}}\)은 branch-dev에서 고정한다. 같은 상수가 §5의 floor 강도 \(\beta\)를 정하는 입력으로도 쓰인다.

### Panel coverage와 새로운 branch

Frozen branch space는 checkpoint 간 직접 비교를 가능하게 하지만, RL 과정에서 discovery에 없던 새로운 reasoning route가 생길 수도 있다. 따라서 기존 cluster에 안정적으로 배정된 rollout의 비율을 **assigned coverage**로 기록하고, 나머지는

\[
M_{\mathrm{new}}
=
P(\text{new or unassigned branch})
\]

로 별도 기록한다. \(M_{\mathrm{new}}\)는 별도의 headline metric이 아니라 frozen panel이 여전히 충분한지를 확인하는 진단값이다.

Branch Breadth는 안정적으로 배정된 branch mass 안에서 \(E\)를 다시 정규화해 계산한다. Assigned mass를 \(m_{\mathrm{asg}}=\sum_jE_j\)라고 하면

\[
\widetilde E_j
=
\frac{E_j}{m_{\mathrm{asg}}}
\]

를 사용한다. Assigned coverage가 branch-dev에서 정한 허용 범위보다 지나치게 낮은 checkpoint에서는 branch-level summary를 보수적으로 해석하고, 새로운 branch mass를 따로 보고한다.

## 3.2 Token-level uncertainty \(H\)

Branch-level exploration과 비교하기 위한 기준선으로 token entropy를 측정한다. 각 reasoning position에서

\[
H_t
=
-\sum_v p(v\mid s_t)\log p(v\mid s_t)
\]

를 계산하고 response 전체의 평균과 위치별 분포를 기록한다.

높은 entropy 자체가 목표는 아니다. \(H\)는 policy가 token 수준에서 얼마나 stochastic한지를 보여주는 익숙한 기준선이다. 우리가 확인하려는 것은 token entropy가 비슷하거나 함께 낮아지더라도 실제 reasoning path의 폭은 다를 수 있는가이다. 대부분의 token에서는 매우 confident하면서도, 풀이 방향이 갈리는 몇몇 지점에서는 여러 reasoning alternative를 유지할 수 있기 때문이다.

## 3.3 Branch entry distribution \(E\)

고정 branch panel 위에서 policy probability가 reasoning branch 사이에 어떻게 배분되는지를 \(E\)로 측정한다. Checkpoint \(\theta\)를 평가할 때 branch-test의 동일 state \(s\)에서 downstream RL과 같은 canonical sampler로 rollout을 생성하고, 각 rollout의 초기 reasoning segment를 frozen cluster에 배정한다.

Branch \(j\)로 들어간 rollout의 비율을

\[
E_{\theta,j}(s)
=
P_\theta(C_{s,j}\mid s)
\]

로 정의한다. 예를 들어 pre-distillation anchor에서 세 branch의 비율이

\[
(0.45,\ 0.35,\ 0.20)
\]

이었는데 distillation 이후

\[
(0.90,\ 0.08,\ 0.02)
\]

가 되었다면, reasoning probability가 하나의 경로에 크게 몰린 것이다.

즉 \(E\)는 하나의 점수가 아니라 branch 위의 probability distribution이다. 이를 통해 특정 inherited branch가 distillation 중 얼마나 약해지거나 사실상 사라졌는지를 직접 추적할 수 있다.

### Prefill 채점: \(E\)의 하한을 직접 재는 보완 측정

\(E\)는 생성 기반이므로 rollout을 cluster에 배정하는 단계를 거치고, 그 배정은 embedding 공간의 성질에 노출된다. 따라서 배정을 거치지 않는 측정을 함께 사용한다. discovery에서 확보한 각 continuation을 고정된 문자열로 두고, checkpoint에 prefix와 함께 넣어 그 문자열의 log-probability를 직접 잰다.

\[
L_\theta(r) = \log P_\theta(r\mid s)
\]

이 값은 \(E\)와 같은 양이 아니다. \(E_{\theta,j}\)는 branch \(C_{s,j}\)에 속하는 모든 표현의 확률을 합한 것인데, 그 표현은 무한히 많고 우리가 가진 것은 유한한 표본뿐이다. 따라서 \(L_\theta\)는 그 합의 일부만 세는 셈이고, \(E_{\theta,j}\)의 하한이 된다. 구체적으로는 checkpoint가 같은 reasoning을 우리 표본에 없는 표현으로 수행할 때 \(L_\theta\)는 떨어지지만 \(E\)는 유지된다. 그러므로 \(L_\theta\)의 하락은 branch 소멸의 증거가 아니라 상한선 없는 신호이며, branch가 실제로 죽었다는 결론은 생성 기반 \(E\)와 함께 볼 때만 내린다.

그럼에도 이 측정을 쓰는 이유는 세 가지다. 첫째, 모든 checkpoint가 정확히 같은 문자열을 채점하므로 배정 단계가 사라지고 문체가 개입할 자리가 없다. 둘째, 제안 방법의 보장 \(q^*(v)\ge\beta\,\pi_A(v)\)가 확률 수준의 진술이므로 \(L_\theta\)가 그 보장과 같은 단위로 검증한다. 셋째, Monte Carlo 오차가 없다. 생성 기반 \(E\)로 확률 0.001짜리 branch를 관측하려면 rollout이 수천 개 필요하지만, 채점은 그 branch의 문자열 하나만 있으면 값을 준다. 정확히 이 희귀 영역이 hard entry loss가 사는 곳이다.

\(L_\theta\)는 raw log-probability로 다루고 정규화하지 않는다. 그리고 서로 다른 continuation의 \(L_\theta\)를 직접 비교하지 않는다. 문체가 그 비교를 지배하기 때문이며, 측정에 쓰는 양은 언제나 같은 continuation에 대한 두 checkpoint의 차이 \(L_{\theta_1}(r)-L_{\theta_2}(r)\)다. 이 차이에서는 continuation 고유의 문체와 난이도가 양쪽에 공통으로 작용해 소거된다. 상세와 실측은 Cal_E_Before_train.md에 있다.

### Soft entry loss와 hard entry loss

모든 \(E\) 감소가 같은 의미를 갖는 것은 아니다. **Soft entry loss**는 branch의 발생 빈도가 크게 줄었지만 주어진 RL rollout budget 안에서는 여전히 일정 확률로 발견되는 경우다. 반면 **hard entry loss**는 branch probability가 너무 작아 실제 rollout budget 안에서는 사실상 관찰되지 않는 경우다.

이 구분은 임의의 raw-probability threshold가 아니라 실제 downstream budget을 기준으로 한다. Branch \(j\)의 entry probability가 \(E_j\)이고 한 state에서 \(G\)번 sampling할 기회가 있다면, 적어도 한 번 해당 branch를 볼 확률은

\[
P_{\mathrm{hit},j}(G)
=
1-(1-E_j)^G
\]

이다. Soft/hard를 나누는 구체적인 hit-probability 기준은 branch-dev에서 고정한다. 이후 recovery 분석에서는 temperature 조절이나 reheating이 soft loss는 복구하면서 hard loss에는 거의 영향을 주지 않는지 확인한다.

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

\[
E\downarrow,\qquad V\approx\text{constant}
\]

라면 branch의 발생 빈도는 줄었지만 자연스럽게 그 branch가 나왔을 때의 성공률은 크게 변하지 않은 것이다. 반대로

\[
E\approx\text{constant},\qquad V\downarrow
\]

라면 branch 자체는 계속 선택되지만 해당 route에서의 성공 가능성이 낮아진 것이다.

### Forced-prefix probe: 선택적 causal check

자연 rollout에서 측정한 \(V\)는 observational statistic이다. 특히 \(E_j\)가 매우 작은 branch에서는 표본 수가 부족할 수 있다. 따라서 중요한 희귀 inherited branch에 한해서는 해당 branch의 짧은 entry prefix를 강제로 넣은 뒤 continuation을 반복 생성하는 별도 probe를 사용한다.

이 실험은 \(V\)의 기본 정의를 바꾸기 위한 것이 아니다. 자연 rollout에서는 branch가 거의 사라졌더라도, 그 reasoning route를 실행할 capability 자체는 남아 있는가를 확인하기 위한 causal check다. 따라서 "distillation이 capability를 지운 것이 아니라 access를 지웠다"와 같은 강한 주장은 자연 \(E,V\) 통계와 forced-prefix 결과가 같은 방향을 보일 때만 사용한다.

## 3.5 모델 수준의 reasoning exploration

Branch별 \(E_j\)와 \(V_j\)는 변화의 원인을 분석하는 데 적합하지만, 여러 checkpoint를 한눈에 비교하기에는 복잡하다. 따라서 opening state를 기준으로 두 개의 요약지표를 사용한다.

### Effective Branch Breadth

Assigned branch distribution \(\widetilde E\)가 실질적으로 몇 개의 reasoning route에 걸쳐 있는지를

\[
B(s)
=
\exp\left(
-\sum_j\widetilde E_j(s)\log \widetilde E_j(s)
\right)
\]

로 계산한다. 이를 **Effective Branch Breadth**라고 부른다.

한 branch에 거의 모든 probability가 몰려 있으면 \(B\)는 1에 가까워지고, 여러 branch를 비슷한 빈도로 사용하면 값이 커진다. 여러 held-out 문제를 비교할 때는 opening state의 \(B(s)\)를 문제별로 계산한 뒤

\[
\overline B
=
\frac{1}{N}\sum_{i=1}^{N}B(s_i)
\]

를 사용한다. Internal fork의 breadth는 mechanistic analysis로 따로 보고하며 모델의 headline breadth에는 섞지 않는다.

### Budgeted Successful Branch Discovery

Reasoning branch가 많더라도 모두 실패한다면 RL에는 큰 도움이 되지 않는다. Branch \(j\)에서 성공 trajectory가 나올 확률은

\[
q_j
=
E_jV_j
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

를 사용한다. Frozen panel 밖에서 새 branch가 많이 생긴 checkpoint에서는 \(M_{\mathrm{new}}\)를 함께 보고, \(D_{\mathrm{succ}}\)를 기존 branch space에 대한 보수적인 추정으로 해석한다.

## 3.6 Base → Distill → RL의 궤적과 RL 학습 신호

모든 checkpoint는 같은 branch-test panel에서 평가한다. 기본 궤적은

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

모델 수준의 대표 비교에는 **Token Entropy \(H\)**, **Effective Branch Breadth \(\overline B\)**, **Budgeted Successful Branch Discovery \(\overline D_{\mathrm{succ}}(G)\)**를 사용한다. Branch별 \(E_j\)와 \(V_j\)는 이 차이가 어디에서 생겼는지 설명하는 mechanistic analysis에 사용한다.

Base, Distillation, RL이 반드시 정해진 방향으로 움직인다고 가정하지는 않는다. 우리가 확인하려는 것은 비슷한 immediate performance와 낮은 token entropy를 가진 모델 사이에서도 branch breadth와 successful discovery가 크게 달라질 수 있는지, 그리고 그 차이가 fixed-budget RL 결과로 이어지는지다.

### 기존 predictor 대비 추가 설명력

Branch-level 지표가 유용하려면 기존의 강한 pre-RL predictor가 이미 설명하는 내용을 다시 재는 데 그쳐서는 안 된다. Quagmires가 제시한 **held-out loss**와 **Pass@large-\(k\)**를 기준 predictor로 두고, 각 distillation recipe와 seed에서 이 값들과 \(H\), \(\overline B\), \(\overline D_{\mathrm{succ}}(G)\)를 함께 기록한다.

여기서 주된 질문은 \(\overline D_{\mathrm{succ}}(G)\)가 Pass@large-\(k\)를 단순히 대체하느냐가 아니다. held-out loss와 Pass@large-\(k\)를 이미 알고 있는 상태에서도 branch-level 정보가 post-RL outcome을 추가로 설명하는가를 본다. 따라서 baseline predictor만 사용한 경우와 여기에 \(\overline B\) 또는 \(\overline D_{\mathrm{succ}}(G)\)를 추가한 경우의 held-out prediction과 rank correlation을 비교한다. 특히 Pass@large-\(k\)와 immediate performance가 비슷하지만 \(\overline D_{\mathrm{succ}}(G)\)가 다른 checkpoint pair가 이후 RL에서도 같은 순서로 갈리는지를 중요하게 본다.

이 분석에서 독립 표본의 기본 단위는 distillation recipe × seed run으로 둔다. 같은 run의 intermediate checkpoint 여러 개를 서로 독립된 모델처럼 세어 predictor 성능을 부풀리지 않는다. \(\overline B\)는 주로 exploration 구조를 설명하는 mechanistic summary로, \(\overline D_{\mathrm{succ}}(G)\)는 fixed rollout budget과 직접 대응하는 predictor candidate로 다룬다.

### Token-level floor가 branch access로 이어지는지 검증

제안 방법은 token probability에 직접 개입하지만, 우리가 중요하게 보는 결과는 branch-level \(E_j\)다. 따라서 token-level protection이 실제 branch entry의 회복으로 이어진다는 연결을 별도로 검증한다.

Inherited branch \(C_{s,j}\)마다 discovery 단계에서 얻은 대표적인 entry segment를 사용한다. 사슬 가운데의 student entry probability는 §3.3의 prefill 채점 \(L_\theta\)로 재며, 이때 두 arm이 같은 segment 집합을 채점받도록 한다. 먼저 그 segment에서 relative floor가 실제로 얼마나 자주 bind하는지, 그리고 teacher target 대비 얼마만큼 probability를 들어 올리는지 기록한다. 측정 창은 branch clustering에 사용한 continuation 길이와 같게 두고, 창 안에서 binding이 어느 위치에 몰리는지도 함께 본다. 진입이 확정되기 전 구간의 binding 수는 floor 강도를 branch 진입 확률로 환산할 때 사용하고, 창 뒷부분의 binding은 branch 내부 execution의 보존을 보는 진단으로 구분한다. 다음으로 protected student가 vanilla student보다 같은 entry segment에 더 높은 probability를 주는지 확인한다. 마지막으로 canonical sampler에서 해당 branch의 실제 entry frequency \(E_j\)가 얼마나 회복됐는지를 측정한다.

즉 branch별로 다음 세 단계가 같은 방향으로 정렬되는지를 본다.

\[
\text{floor binding / target lift}
\rightarrow
\text{student entry probability lift}
\rightarrow
\Delta E_j
\]

Floor가 거의 bind하지 않은 inherited branch는 자연스러운 negative control이 된다. 이 branch들에서는 student entry probability와 \(E_j\)도 크게 변하지 않아야 한다. 반대로 floor가 강하게 작동한 branch일수록 \(E_j\)가 더 많이 회복된다면, relative-floor가 단순한 token-level regularizer가 아니라 실제 reasoning-mode accessibility를 복원한다는 근거가 된다.

### RL 학습 신호와의 연결

Branch-level exploration이 실제 RL optimization으로 이어지는 과정도 함께 기록한다. GRPO group에서 모든 rollout이 실패한 **all-fail group**, 모든 rollout이 성공한 **all-success group**, 성공과 실패가 함께 존재하는 **informative group**의 비율을 측정한다. 전체적으로 검증하려는 mechanistic chain은 다음과 같다.

\[
\text{inherited branch entry에서 floor가 작동}
\rightarrow
\text{student entry probability 증가}
\rightarrow
E_j\text{ 회복}
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

각 intervention이 끝난 뒤에는 sampling 설정을 다시 동일한 canonical sampler로 되돌린다. 모델 수준에서는 \(H\), \(\overline B\), \(\overline D_{\mathrm{succ}}(G)\)를 중심으로 비교하고, 필요한 경우 branch별 \(E_j\)와 \(V_j\)를 이용해 변화의 원인을 분해한다. 이렇게 해야 "temperature를 높여둔 동안만 다양해진 것"과 "policy 자체의 exploration structure가 실제로 회복된 것"을 구분할 수 있다.

Recovery는 두 조건을 함께 본다. 첫째는 compute다. 같은 총 post-handoff compute에서 얼마나 복구되는지를 비교한다. 둘째는 information access다. 현재 damaged checkpoint와 reward만 사용하는지, 자기 logits가 필요한지, pre-distillation anchor나 teacher information을 다시 사용해야 하는지를 구분한다.

따라서 어떤 손상이 절대적으로 irrecoverable하다고 주장하지 않는다. 대신 다음을 묻는다.

> **같은 budget과 information access에서 어떤 손상은 쉽게 돌아오고, 어떤 손상은 그렇지 않은가?**

특히 \(H\)는 회복됐는데 \(\overline B\)는 돌아오지 않는지, \(\overline B\)는 회복됐는데 \(\overline D_{\mathrm{succ}}\)는 돌아오지 않는지, 그리고 단순히 RL compute를 더 주는 것보다 과거 anchor information을 다시 쓰는 편이 더 효과적인지를 비교한다.

---

# 5. 분석에서 도출되는 개선 방법

## 5.1 Relative-floor projection

분석에서 vanilla distillation이 inherited branch의 \(E_j\)를 과도하게 낮추고, 그 결과 \(\overline B\)와 \(\overline D_{\mathrm{succ}}\)가 떨어진다는 현상이 확인되면 entry preservation을 적용한다.

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

Teacher와 anchor를 전역적으로 섞는 arithmetic mixture와 달리, 실제 floor를 위반한 candidate만 수정한다. 이 token-level 개입이 실제 branch access를 복구하는지는 §3.6의 branch-entry 분석에서 floor binding → student entry probability → \(E_j\) 변화의 순서로 직접 확인한다.

## 5.2 \(V\)가 독립적으로 무너지는 경우

주 방법은 우선 branch entry preservation에 집중한다. 파일럿에서 비슷한 \(E\)를 가진 branch들 사이에서도 distillation 이후 \(V\)가 반복적으로 크게 낮아지고, 이 변화가 post-RL 성능을 추가로 설명한다면 그때만 short-horizon continuation preservation을 두 번째 방법으로 확장한다.

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
