# Cost(beta): how much does the state distribution matter?

On-policy distillation visits states the student generates. At the start the student is the anchor; as training proceeds it drifts toward the target. The two columns bracket that drift.

| beta | anchor states (t=0) | teacher states (upper bracket) | ratio | toy |
|---|---|---|---|---|
| 0.1 | 0.019 | 0.141 | 7.5x | 0.004 |
| 0.2 | 0.045 | 0.307 | 6.8x | 0.026 |
| 0.4 | 0.114 | 0.674 | 5.9x | 0.100 |
| 0.8 | 0.308 | 1.510 | 4.9x | 0.370 |

At the canonical beta=0.4 the bracket is [0.114, 0.674] nats, i.e. the target sits between 1.12x and 1.96x off the teacher per token depending on how far the student has drifted.

Per-kind means (unweighted), showing where the gap comes from:

| state kind | anchor states | teacher states | ratio |
|---|---|---|---|
| opening | 11.761 | 11.751 | 1.0x |
| early | 0.540 | 4.417 | 8.2x |
| internal | 0.090 | 0.683 | 7.6x |

The opening state is literally the same state in both runs (only the prompt), so it matches; the gap is entirely in the generated region. Median internal cost is 0.0018 on anchor states versus 0.0846 on teacher states.
