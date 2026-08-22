# Branch panel과 \(E\) 측정

이 문서는 reasoning branch panel을 만들고 checkpoint별 branch entry distribution \(E\)를
측정하는 방법의 정본이다. API 호출법과 파일 형식은
[BranchDev README](../Experiment/BranchDev/README.md)에 따로 적는다.

## 1. 측정 대상

Branch는 한 문제를 푸는 **주된 수학적 접근법**이다. 문투, 첫 문장, 특정 token은 branch가
아니다. 이들은 접근법으로 들어가는 entry prefix가 될 수는 있지만, branch의 이름은
`cauchy`, `trig-param`, `geometric`처럼 풀이의 중심 원리를 가리킨다.

응답 하나에는 primary approach 하나만 붙인다. Judge가 하나를 고르기 어려우면
`ambiguous`, 기존 목록에 없는 명료한 방법이면 `other`, 추론이 무너지거나 풀이로 볼 수
없으면 `failed`로 둔다. 여러 방법이 섞인 응답도 결론을 실제로 운반한 방법 하나로 hard
classification한다.

Panel은 문제별로 만든다. 문제 180의 `cauchy`와 문제 115의 `cauchy`가 같은 이름을 가질
수는 있지만, \(E\)는 각 문제의 고정된 branch 목록 위에서 계산한다.

## 2. Branch universe를 만드는 두 경로

가능한 접근법을 넓게 찾는 일과 Base가 실제로 열어 둔 entry를 찾는 일은 성격이 다르다.
그래서 API와 Base를 한 종류의 생성원으로 취급하지 않는다.

### 2.1 API 완결 풀이

API 모델은 문제를 처음부터 끝까지 푼다. 서로 다른 여섯 계열을 쓰는 이유는 같은 모델의
표본 수나 temperature보다 모델 간 차이가 접근법의 폭을 더 크게 늘렸기 때문이다.

| 모델 | 고정 endpoint |
|---|---|
| GLM 5.2 | `ambient/fp8` |
| DeepSeek V4 Flash 0731 | `deepinfra/fp8` |
| Qwen3.8-27B | `chutes/fp8` |
| MiniMax M3 | `deepinfra/fp8` |
| MiMo v2.5 Pro | `deepinfra/fp8` |
| Muse Glimmer 30B | `deepinfra/bf16` |

문제당 모델별 16개를 생성한다. endpoint는 고정하고 fallback은 허용하지 않는다. 최종 panel에
들어가는 자료는 완결 풀이이며, 기존의 256-token discovery trajectory는 source 선정과 초기
approach 목록을 만든 pilot 자료로만 쓴다.

완결 응답은 기존 수학 채점기로 정답 여부를 확인한다. 정답인 응답에 대해 Judge가 primary
approach를 판정한다.

### 2.2 Base recursive tree

Base는 완결 답안을 만들지 않는다. Base의 역할은 자기 분포 안에 있는 partial reasoning
path를 넓게 전개하는 것이다.

Opening에서는 실측으로 확인한 1% 기준을 그대로 쓴다. 첫 token에서 확률 1% 이상인 후보를
모두 열고, 각 후보의 다음 위치에서도 1% 이상인 후보를 모두 연다. 이 특례는 낮은 확률의
opening이 오히려 서로 다른 방법으로 들어가는 경우를 놓치지 않기 위해 둔다.

그 뒤에는 Base top-1을 따라가다가 internal fork를 만났을 때 다시 분기한다. Internal fork는
다음 두 조건을 함께 만족해야 한다.

1. next-token entropy가 기준 이상이다.
2. 확률 1% 이상인 token이 둘 이상이다.

현재 기본값은 entropy 1.5, depth 3, 다음 fork를 찾는 최대 길이 48 token이다. Opening의
두 위치를 depth 1로 세고, internal fork를 열 때마다 depth가 하나씩 늘어난다. Tree 생성기는
partial prefix와 확률만 저장한다. Judge, Qwen3-14B, 정답 판정은 호출하지 않는다.

Leaf의 확률은 forced token과 top-1으로 걸은 token의 확률을 모두 곱한 값이다. 같은 tree의
최종 leaf들은 서로 겹치지 않으므로, 같은 semantic approach로 판정된 leaf들의 확률을
합산할 수 있다.

## 3. Base leaf validation

Base leaf가 문법 파편인지, 실제 풀이로 이어질 수 있는 entry인지 별도 단계에서 확인한다.

먼저 Judge가 Base partial prefix만 보고 provisional primary approach를 판정한다. Prefix가
아직 어느 방법인지 드러내지 못하면 `ambiguous`로 둔다.

그다음 Qwen3-14B가 같은 prefix에서 네 번 이어 쓴다. 각 completion에는 두 판정을 붙인다.

- 기존 수학 채점기로 정답인가?
- Judge가 보기에 prefix의 접근법을 유지했는가?

두 조건을 동시에 만족하는 completion이 네 번 중 하나라도 있으면 leaf를 viable로 본다.
이 기준은 성공률을 재기 위한 것이 아니다. 해당 entry가 강한 executor 아래에서 같은 방법을
유지하며 유효한 풀이로 발전할 수 있는지만 확인한다.

Qwen3-14B가 정답을 맞혔더라도 prefix의 방법을 버리고 다른 방법으로 다시 풀었다면 통과시키지
않는다. 반대로 viable completion이 하나 있으면 나머지 세 번의 실패 때문에 leaf를 버리지
않는다.

## 4. 하나의 semantic panel로 합치기

정답인 API 풀이와 validated Base leaf record를 한데 모은다. Base record에는 partial prefix와
validation을 통과한 Qwen3-14B completion을 함께 남긴다. Source는 metadata로만 보존한다.

Judge는 먼저 문제별 candidate approach 목록을 만든다. 같은 방법의 다른 이름은 하나로
정리한다. 목록이 정해지면 API 응답과 Base branch를 그 목록으로 다시 hard classification한다.
API와 Base가 같은 primary approach를 사용했다면 같은 branch에 들어간다.

이 분류에서 `other`로 반복되는 명료한 접근법이 있으면 candidate 목록에 추가하고 다시 분류한다.
새 접근법이 거의 남지 않을 때 panel을 동결한다. 별도의 수치형 coverage threshold는 미리 두지
않는다. 동결 이후 처음 나타나는 명료한 접근법은 `other`로 유지한다.

이 단계가 끝나면 문제마다 다음 자료가 남는다.

- 고정된 semantic branch 목록
- branch별 API 완결 응답
- branch별 validated Base entry prefix
- 각 prefix의 token IDs와 Base path probability
- `other`, `ambiguous`, `failed` 자료

Panel을 만든 뒤 checkpoint마다 branch 목록을 다시 만들지 않는다. 이후 모델은 같은 문제와
같은 branch 목록 위에서만 비교한다.

## 5. 두 가지 \(E\)

### 5.1 Generated branch occupancy

Checkpoint \(\theta\)에서 문제 \(s\)의 응답을 \(N\)개 생성하고 Judge가 각 응답을 hard
classification한다.

\[
E^{\mathrm{gen}}_{\theta,j}(s)
=
\frac{n_{\theta,j}(s)}{N}
\]

분모는 전체 생성 수다. Semantic branch뿐 아니라 `other`, `ambiguous`, `failed`도 같은
분모에 둔다.

\[
\sum_j E^{\mathrm{gen}}_{\theta,j}
+E_{\mathrm{other}}
+E_{\mathrm{ambiguous}}
+E_{\mathrm{failed}}
=1
\]

이 값은 실제 sampler가 어떤 접근법을 얼마나 자주 꺼내는지를 보여 준다.

### 5.2 Entry accessibility

Base에서 검증된 branch \(j\)의 entry prefix 집합을 \(\mathcal G_j\)라고 하자. 같은
prefix들을 모든 checkpoint가 채점한다.

\[
E^{\mathrm{entry}}_{\theta,j}(s)
=
\sum_{g\in\mathcal G_j}P_\theta(g\mid s)
\]

Base tree의 leaf 집합은 prefix-free이므로 중복 없이 합할 수 있다. 이 값은 branch 전체에
놓인 모든 표현의 확률이 아니라, panel이 확보한 entry를 통해 직접 재는 covered entry mass다.
그래서 생성 기반 \(E\)를 대신하지 않고 accessibility를 설명하는 보완 측정으로 쓴다.

API-only branch의 \(E^{\mathrm{entry}}\) 구성법은 아직 고정하지 않았다. API 완결 풀이에서
접근법이 명료해지는 대표 prefix를 추출하는 방안은 실제 panel을 본 뒤 결정한다. 그전까지는
해당 값을 N/A로 두되, 생성 기반 \(E^{\mathrm{gen}}\)에는 branch를 그대로 포함한다.

### 5.3 두 측정의 판독

| Entry accessibility | Generated occupancy | 판독 |
|---|---|---|
| 높음 | 높음 | 접근 가능하고 실제로 사용한다 |
| 높음 | 낮음 | 접근은 가능하지만 sampler가 잘 선택하지 않는다 |
| 낮음 | 높음 | panel 밖 entry나 새로운 표현을 사용했을 수 있다 |
| 낮음 | 낮음 | 해당 branch의 접근이 실질적으로 줄었다 |

## 6. Branch breadth와 성공 branch

Effective Branch Breadth는 정상 semantic branch의 생성 질량 안에서
\(E^{\mathrm{gen}}\)을 다시 정규화한 뒤 계산한다. Assigned semantic mass를
\(m_{\mathrm{asg}}=\sum_jE^{\mathrm{gen}}_j\)라고 하면

\[
\widetilde E_j=\frac{E^{\mathrm{gen}}_j}{m_{\mathrm{asg}}},
\qquad
B(s)=\exp\left(-\sum_j\widetilde E_j\log\widetilde E_j\right).
\]

Breadth만 보고 모델을 평가하지 않는다. Assigned mass, `other`, `ambiguous`, `failed`를 함께
보고해야 실패가 늘어난 모델을 넓거나 좁은 모델로 잘못 읽지 않는다.

Branch별 조건부 성공률 \(V_j\)가 준비되면

\[
q_j=E^{\mathrm{gen}}_jV_j,
\qquad
D_{\mathrm{succ}}(s;G)
=
\sum_j\left[1-(1-q_j)^G\right]
\]

를 계산한다. Panel validation에서 Qwen3-14B가 한 번 성공했다는 사실은 \(V_j\)가 아니다.
\(V_j\)는 평가하려는 checkpoint 자신이 해당 branch를 실행했을 때의 성공률이다.

## 7. 시각화

주 그림은 문제별 small multiple이다. 한 행은 한 문제이고, 열은 Base, vanilla Distill,
EMBER, vanilla→RL, EMBER→RL checkpoint다. 같은 행의 모든 열은 동일한 semantic atlas를
공유한다.

검증하려는 예상 패턴은 Base가 넓은 접근법 범위를 보이고, vanilla Distill에서 범위가 좁아진
뒤 vanilla→RL에서 더 국소화되는 것이다. EMBER와 EMBER→RL은 같은 성능대의 vanilla arm보다
더 많은 접근법을 유지해야 한다. 이는 그림에 미리 심는 구조가 아니라 데이터가 반증할 수 있는
가설이다.

점 하나는 생성 응답 하나이며, Judge가 붙인 primary branch 영역에 놓인다. 영역의 점 개수와
밀도는 \(E^{\mathrm{gen}}\)에 대응한다. \(E^{\mathrm{entry}}\)가 정의된 branch에는 그 값을
branch 옆의 수치나 작은 막대로 함께 표시한다.

Raw text embedding은 branch를 정하거나 checkpoint 응답을 배정하는 데 쓰지 않는다. 기존
실측에서 같은 방법을 쓴 다른 모델의 응답보다 다른 방법을 쓴 같은 모델의 응답이 더 가까웠기
때문이다. 필요하면 semantic branch 영역 안에서 점을 흩뜨리는 보조 표현으로만 사용한다.

이 그림은 token entropy의 대용물이 아니다. EMBER가 보존하려는 것은 높은 entropy 자체가 아니라
서로 다른 풀이 입구에 접근할 가능성이다. 따라서 \(H\)가 낮거나 vanilla arm과 비슷해도
\(E^{\mathrm{gen}}\)과 \(E^{\mathrm{entry}}\)가 더 넓게 남을 수 있으며, 바로 그 분리를 보여 주는
것이 panel의 목적이다.

## 8. 기존 실측이 남긴 결정

초기 discovery는 MATH 20문제에서 여섯 API 모델을 각각 16회 호출했다. 1,920건 중 1,913건이
성공한 partial trajectory다. 현재 진단 산출물에서 Qwen3-Embedding-8B의 source-only cosine은
0.9177, method-only cosine은 0.8952였고, 문제별 비교에서도 17문제 중 15문제에서 source
효과가 더 컸다. 이 결과로 raw embedding clustering을 branch 정의에서 제외했다.

Base habitat에서는 20문제, temperature 0.7과 1.0, 문제당 24회씩 총 960개를 라벨했다.
문제당 방법 수 중앙값은 3이었고, 0.05~0.2 질량의 소수 방법도 실제 정답에 도달했다. 방법이
갈리고 Base가 실행할 수 있었던 8문제를 branch panel 후보로 남겼다.

초기 internal-fork probe는 entropy가 큰 붕괴 구간이나 `use`, `proceed` 같은 문법 선택을
잡았다. 반면 opening에서는 1%대의 낮은 확률 후보가 서로 다른 접근으로 이어졌다. 문제 180의
opening 두 위치를 전개한 pilot은 35개 leaf와 65%의 prefix mass를 얻었다. 다만 leaf가
`To find the`, `We need to`처럼 짧아 방법이 하나로 정해지지 않았다. 현재 recursive tree는
이 pilot을 opening stage로 유지하고 다음 internal fork를 더 깊게 전개한다.

Opening cutoff에는 1%를 쓰며, \(m_{\min}=0.10\)은 token cutoff가 아니다. 10%를 token마다
적용하면 서로 다른 접근법을 열던 낮은 확률 후보가 잘렸기 때문이다.

## 9. 현재 실행 순서

1. API 모델에서 문제별 완결 풀이를 생성한다.
2. Base recursive tree로 partial prefix를 만든다.
3. Base leaf를 Judge와 Qwen3-14B로 validation한다.
4. 정답 API 풀이와 validated Base branch를 합쳐 문제별 taxonomy를 만든다.
5. 반복되는 `other`를 새 branch로 반영한 뒤 panel을 동결한다.
6. 고정 panel의 Base-validated branch에서 checkpoint별 entry accessibility를 잰다.
7. 같은 panel에서 checkpoint 응답을 hard classification해 generated occupancy를 잰다.
8. 문제별 small multiple과 집계 지표를 만든다.
