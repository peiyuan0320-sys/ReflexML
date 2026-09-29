<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# ReflexML Gradient-Noise Attenuation v1: frozen primary analysis

## Data integrity

- N = 36 states; 144/144 required cells finite and evaluable; exclusions: none.
- Historical m=1: admitted Momentum-Reset A1 Reset cells only; admission SHA-256 `5691e4637496bf5906c8d5b9927811947eb31cc9d3a4d65a0126289ba6b2a916`.
- New m=4: 72 exact production terminals at `internal-artifacts/reflexml-gradient-noise-v1-production-20260927-001`; official checker PASS, no unresolved or extra records.
- State is the inferential unit. All intervals use the same 10,000 x 36 PCG64 index matrix.

## Primary validation-loss results

| Estimand | Point estimate | 95% percentile CI |
|---|---:|---:|
| Delta_1 | +0.048418212 | [+0.036217874, +0.062439636] |
| Delta_4 | +0.011728252 | [+0.009196234, +0.014518654] |
| Psi_4 | +0.036689960 | [+0.024364726, +0.050673242] |
| R_10 | +0.056532947 | [+0.045021568, +0.069439837] |
| R_05 | +0.019842988 | [+0.016204351, +0.023923995] |

Positive Delta favors LR=0.05 (lower loss). Positive R means m=4 lowers loss at that LR.

Four cell means (loss): m1_lr10=0.463011822, m1_lr05=0.414593610, m4_lr10=0.406478874, m4_lr05=0.394750622.

## Primary conclusion

m=4 attenuates the LR=0.05 validation-loss advantage relative to m=1. This establishes causal sensitivity of the LR effect to the whole m=4 intervention and is compatible with a stochastic-gradient contribution, without identifying it uniquely.

## Persistence under m=4

The LR=0.05 validation-loss advantage persists under m=4. The amount/type of minibatch-gradient variability attenuated by m=4 is not necessary for the existence of the immediate LR advantage; this does not make stochastic mechanisms irrelevant.

## Cell movement

- LR=0.10: m=4 lowers validation loss; +0.056532947 | [+0.045021568, +0.069439837].
- LR=0.05: m=4 lowers validation loss; +0.019842988 | [+0.016204351, +0.023923995].

The statewise and aggregate identity Psi_4 = R_10 - R_05 passed at absolute tolerance 1e-12.

## Secondary validation accuracy

| Estimand | Point estimate | 95% percentile CI |
|---|---:|---:|
| Delta_1 | -0.014972222 | [-0.018883472, -0.011533194] |
| Delta_4 | -0.002050000 | [-0.003066667, -0.001100000] |
| Psi_4 | -0.012922222 | [-0.016872361, -0.009400000] |
| R_10 | -0.018622222 | [-0.022300000, -0.015294444] |
| R_05 | -0.005700000 | [-0.007161111, -0.004355556] |

Accuracy is supporting only. The negative Delta estimates favor LR=0.05 in accuracy; negative R means m=4 raises accuracy at that LR. The negative accuracy interaction indicates attenuation of that accuracy advantage under m=4, without changing the primary loss conclusion.

## Mechanism consequence

Inherited S17 momentum was tested separately in the Momentum-Reset experiment. Here the positive Psi_4 supports sensitivity to the compound m=4 gradient-averaging intervention, while positive Delta_4 shows that the LR advantage persists under it. The findings are compatible with a stochastic-gradient contribution, but deterministic finite-step dynamics and stochastic or interaction explanations remain possible; m=4 does not isolate noise.

## Reproducibility and procedure

- Analysis JSON: `internal-artifacts/primary_analysis.json`; SHA-256 `46f485320ed703d0a2651b7bf8ecce18eb515d0449eaf58415c99ec6d17ddb83`; 90894 bytes.
- Bootstrap matrix: `internal-artifacts/bootstrap_state_indices.npy`; SHA-256 `38f22b2024d417b3f7af3306ea49fb30d8de4638939e1cb936a4123f58e56003`; 2880128 bytes; NumPy 2.4.4.
- Protocol timing deviation: one m=4 terminal endpoint was previewed during schema inspection before the common bootstrap matrix was materialized. Thus the pre-outcome materialization requirement was not fully met. The seed, draw rule, and estimands were prospectively frozen and unchanged; the numerical analysis is reproducible but full procedural compliance is not claimed.

## Frozen scientific conclusion

m=4 attenuates the LR=0.05 validation-loss advantage relative to m=1. This establishes causal sensitivity of the LR effect to the whole m=4 intervention and is compatible with a stochastic-gradient contribution, without identifying it uniquely. The LR=0.05 validation-loss advantage persists under m=4. The amount/type of minibatch-gradient variability attenuated by m=4 is not necessary for the existence of the immediate LR advantage; this does not make stochastic mechanisms irrelevant.
