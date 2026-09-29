<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# ReflexML mechanism-stage closure: epoch18 LR switch and inherited momentum

**Scope:** Independent scientific synthesis of completed, frozen LR-Switch v1 and Momentum-Reset v1 results. This document introduces no new estimand, analysis, intervention, or experiment. It does not amend either protocol or any result authority.

## Evidence chain and interpretation boundary

1. The frozen Phase 5 branch diagnostic describes the h=3→4 attenuation as predominantly Wait3-side catch-up: mean Now validation loss changed `0.412646266 → 0.414629081` (`+0.001982815`), while mean Wait3 validation loss changed `0.459700126 → 0.414719332` (`−0.044980794`). All 36 states had a negative Wait3 change. These are historical descriptive observations, not the LR-Switch causal contrast. The deterministic summary at `post_phase5_branch_diagnostic/summary.json` confirms the changes and state sign count; the absolute levels above are supplied in the synthesis brief.
2. LR-Switch v1 held the historical Wait3 S17 model parameters, optimizer state, inherited momentum, and paired epoch18 future minibatches fixed, changing only epoch18 LR (`0.10` Continue versus `0.05` Switch). It used 36 independent states and two replicas per state. Its frozen primary result is in `internal-artifacts/primary_analysis.json` (SHA-256 `9b7d8a940c3b62590cad1793070c523ddb0c06c8592d2abe4d7c84378454e791`). The prespecified secondary result is in `internal-artifacts/secondary_analysis.json` (SHA-256 `463f587b8fe0d61ac92a7f543a3f5f197252e788cad9882ec2f35e49cc87f772`).
3. Momentum-Reset v1 retained the exact historical Keep cells and added Reset cells for the same 36 states and A1/A2 future replicas. Reset set the **numerical values of all inherited S17 momentum buffers to zero**, preserving their entries and optimizer structure; momentum coefficient `0.9` and within-epoch momentum accumulation remained active. The production completion manifest has SHA-256 `69260f4fbad9b2db434520949edb2ab1a6464fdc1e0eeaf1d6e554a15b521efa`. The frozen primary and secondary analysis JSONs have SHA-256 `cc2f3fc59dc36c0fbf3a567e463542595e98766b6fc6f45bcda2a2a21e44bbab` and `3a976649ce4433e4f5ad31b568bbf9aad65578bb03306265e6a00f97b9dcda0e`, respectively. They report 144 valid, evaluable Reset branches, complete four-cell identity and order checks, and no Keep reruns. The historical Keep estimate was reproduced.
4. The frozen LR-Switch and Momentum-Reset intervals are state-level bootstrap intervals; the two protocols use different frozen bootstrap resamples. Their Keep-row point estimates agree, but their Keep-row interval endpoints need not match. Replicas are paired within states and are not 72 independent population states. Epoch18 validation loss is the Momentum-Reset **primary** outcome and the Reset × LR interaction is its **primary estimand**. Training loss and validation accuracy remain secondary.

The factorial arithmetic is `Delta_K = L_10K − L_05K`, `Delta_R = L_10R − L_05R`, `I = Delta_K − Delta_R`, `R_.10 = L_10K − L_10R`, and `R_.05 = L_05K − L_05R`. The same arithmetic applies to every outcome. For loss, positive `Delta` favors LR `0.05`; for accuracy, negative `Delta` favors LR `0.05`.

## A. What is established?

### 1. Direct causal findings at the primary endpoint

Under the controlled LR-Switch intervention, `delta_switch = L_Continue − L_Switch = 0.041100323`, 95% CI `[0.033847031, 0.048712969]`. Reducing epoch18 LR from `0.10` to `0.05` causally lowers immediate validation loss **in this setup**. This is the LR-Switch primary finding, not proof of a particular optimization mechanism.

The Momentum-Reset factorial directly tests inherited S17 momentum. The primary validation-loss results are:

| Estimand | Estimate | 95% state-level CI |
| --- | ---: | ---: |
| `Delta_K` | `0.041100323` | `[0.033806525, 0.048786008]` |
| `Delta_R` | `0.044404765` | `[0.037130751, 0.052416257]` |
| `I` | `−0.003304442` | `[−0.008549983, 0.001892262]` |
| `R_.10` | `−0.003718550` | `[−0.008759467, 0.001241476]` |
| `R_.05` | `−0.000414107` | `[−0.001697230, 0.000822699]` |

`Delta_R` establishes that the immediate LR `0.05` validation-loss advantage survives removal of inherited S17 momentum. The primary interaction interval contains zero, so inherited-momentum modification of that LR effect is **not established**. Neither the interaction nor either fixed-LR Reset effect establishes a direction by its interval. The negative Reset-effect point estimates mean Reset raised mean validation loss in both LR cells, but this point pattern is not a supported directional Reset effect.

### 2. Robust cross-metric factorial pattern

After inherited-momentum Reset, LR `0.05` is favored on all three prespecified outcomes in their correct directions: lower primary validation loss, lower secondary online training loss, and higher secondary validation accuracy. This is cross-metric consistency of the **simple LR effect after Reset**, not a joint test, a new primary endpoint, or proof of one mechanism. No interaction is established for primary validation loss or secondary accuracy.

### 3. Secondary-only findings

The original LR-Switch secondary contrasts were `Continue − Switch = 0.037655069` for online training loss (95% CI `[0.034508023, 0.040958182]`) and `−0.012611111` for validation accuracy (95% CI `[−0.015077778, −0.010127778]`). Their directions favor LR `0.05`.

The Momentum-Reset secondary factorial results are:

| Outcome and estimand | Estimate | 95% state-level CI |
| --- | ---: | ---: |
| Online training loss `Delta_K` | `0.037655069` | `[0.034596227, 0.040966995]` |
| Online training loss `Delta_R` | `0.030754888` | `[0.028224463, 0.033530275]` |
| Online training loss `I` | `0.006900180` | `[0.004886686, 0.009054503]` |
| Online training loss `R_.10` | `0.012582529` | `[0.009685353, 0.015830376]` |
| Online training loss `R_.05` | `0.005682348` | `[0.004375393, 0.007139072]` |
| Validation accuracy `Delta_K` | `−0.012611111` | `[−0.015136181, −0.010227778]` |
| Validation accuracy `Delta_R` | `−0.013683333` | `[−0.016113958, −0.011391597]` |
| Validation accuracy `I` | `0.001072222` | `[−0.000738958, 0.002886181]` |
| Validation accuracy `R_.10` | `0.000911111` | `[−0.000811111, 0.002619444]` |
| Validation accuracy `R_.05` | `−0.000161111` | `[−0.000625069, 0.000322222]` |

Online training loss is the sample-weighted mean of minibatch losses encountered **along** the epoch18 optimization path, evaluated before their corresponding updates; it is not necessarily full-training-set loss at final parameters. For this secondary metric, Reset lowers online training loss at both LRs, more at LR `0.10`, and the positive interaction is established. The LR `0.05` training-loss advantage persists after Reset. For secondary accuracy, a negative `Delta_R` means higher accuracy under LR `0.05`; its interaction and fixed-LR Reset effects are not established.

The historical LR-Switch catch-up decomposition remains distinct: `D = L17 − L_S18 = 0.045849290` (95% CI `[0.036812897, 0.056272963]`), whereas Continue catch-up `kappa_C = 0.004748967` (95% CI `[−0.007571413, 0.016896603]`). The latter is unresolved. The LR switch is not shown necessary for **all** catch-up.

## B. What has been weakened?

The hypothesis that **inherited S17 momentum is the explanation of the immediate epoch18 LR `0.05` endpoint benefit** is weakened in a precise sense. The Reset intervention removes that inherited state, yet the primary LR advantage remains clearly positive. Inherited S17 momentum therefore **cannot by itself explain** the validation-loss response. Separately, the primary factorial test does **not establish** inherited-momentum modification of the LR effect. These statements do not imply a zero interaction: the interaction interval includes both negative and positive values, and no equivalence margin was frozen. They also do not imply that momentum as a whole is irrelevant, because momentum dynamics continued during epoch18 and a secondary online-loss interaction was observed.

## C. What is not ruled out?

- Momentum **newly accumulated during epoch18** remains active under both Keep and Reset; this experiment does not remove or isolate it.
- Deterministic finite-step dynamics remain compatible but were not directly isolated.
- Stochastic finite-step dynamics remain compatible but were not directly isolated.
- Interactions among newly accumulated momentum, deterministic dynamics, and stochastic dynamics remain compatible. None is established by this factorial contrast.

“Smaller LR gives smaller parameter updates” restates part of the intervention; by itself it does not identify the causal dynamics that produce the endpoint validation response.

## D. Cross-metric interpretation

The three outcomes have different measurement targets. Primary validation loss and secondary accuracy evaluate endpoint validation behavior; secondary online training loss averages losses encountered during successive minibatch updates. A Reset × LR interaction on that trajectory-weighted metric can therefore coexist with unresolved interactions on both endpoint validation metrics. This observed heterogeneity supports an **outcome-specific** inherited-state effect and forbids substituting the favorable training-loss interaction for the primary validation-loss interaction. It does not demonstrate that endpoint interactions are absent, nor does it identify why the outcome patterns differ.

The bounded statement “inherited S17 momentum affects the within-epoch optimization trajectory, but current evidence does not establish that it drives the endpoint validation LR advantage” is supported **if** “affects the trajectory” means affects the measured online-loss functional accumulated along that trajectory. The secondary intervention contrast directly supports that narrower reading. It does not establish a specific gradient, parameter-path, overshoot, or mediation mechanism. The second clause follows from the unresolved primary interaction and the persisting Reset-row primary LR advantage.

## E. M / D / N update

| Category | Qualitative status | Reason and limit |
| --- | --- | --- |
| **M: inherited S17 momentum state** | **Weakened as a sufficient endpoint explanation; primary modification unresolved; secondary metric effect established.** | The validation LR advantage survives Reset; primary `I` includes zero. The online training-loss `I` excludes zero. M means only the inherited S17 buffer values, not all momentum dynamics. |
| **D: deterministic finite-step dynamics** | **Remains compatible; not tested directly.** | Neither experiment isolates deterministic dynamics from stochastic dynamics or continuing momentum. |
| **N: stochastic finite-step dynamics** | **Remains compatible; not tested directly.** | Paired future streams control the LR contrast; they do not decompose the response into stochastic and deterministic mechanisms. |

These statuses are not a likelihood ranking.

## Claims table

| Claim | Evidence | Status | Allowed wording | Forbidden overclaim |
| --- | --- | --- | --- | --- |
| LR switch causally improves immediate validation loss. | LR-Switch primary `delta_switch = 0.041100323`, CI wholly positive; controlled paired LR intervention. | **Established in this setup.** | “Switching epoch18 LR from `0.10` to `0.05` lowers immediate validation loss.” | “This identifies the optimization mechanism” or “works generally.” |
| LR `0.05` advantage persists after inherited momentum Reset. | Momentum-Reset primary `Delta_R = 0.044404765`, CI wholly positive. | **Established in this setup.** | “The endpoint validation LR advantage survives removal of inherited S17 momentum.” | “Momentum dynamics are unnecessary.” |
| Inherited S17 momentum modifies the primary validation-loss LR effect. | Primary `I = −0.003304442`, CI `[−0.008549983, 0.001892262]`. | **Not established; unresolved.** | “No primary effect modification was established.” | “The interaction is zero,” “equivalent,” or “Reset proves no modification.” |
| Inherited momentum affects the online training trajectory. | Secondary online-loss `I = 0.006900180`, CI wholly positive; both fixed-LR Reset effects positive. | **Supported for the measured online-loss functional.** | “Reset changes losses accumulated along epoch18 optimization, with an outcome-specific LR interaction.” | “A specific parameter-path or mediation mechanism is proven.” |
| Momentum as a whole is irrelevant. | Reset leaves coefficient `0.9` and newly accumulated momentum active; secondary online-loss effects. | **Not supported.** | “This study does not test the irrelevance of all momentum dynamics.” | “Momentum is irrelevant.” |
| Deterministic finite-step mechanism is established. | No direct D-isolating intervention or frozen decomposition. | **Unresolved; not tested directly.** | “Deterministic dynamics remain compatible.” | “D explains the benefit.” |
| Stochastic finite-step mechanism is established. | No direct N-isolating intervention or frozen decomposition. | **Unresolved; not tested directly.** | “Stochastic dynamics remain compatible.” | “N explains the benefit.” |
| LR switch is necessary for all historical catch-up. | Continue `kappa_C` interval includes zero. | **Not established.** | “Spontaneous Continue catch-up remains unresolved.” | “All catch-up requires the LR switch.” |

## Frozen non-claims

This closure makes **no** equivalence or zero-interaction claim; no “momentum irrelevant” claim; no gradient–momentum alignment, overshoot, Hessian, or sharpness mechanism claim; no claim that a deterministic or stochastic mechanism has been established; no mediation percentage; no external generality or novelty claim; and no claim that the LR switch is necessary for all catch-up. It makes no ranking of M/D/N, no post-hoc primary-outcome change, and no new numerical analysis of the frozen records.

## Remaining highest-value mechanistic uncertainty

Is the surviving immediate epoch18 LR `0.05` endpoint validation advantage primarily attributable to deterministic finite-step dynamics, stochastic finite-step dynamics, or their interaction, potentially alongside momentum accumulated during that epoch?

This is a question only. **No new experiment is designed or authorized by this closure.**

## Scientific closure

The immediate epoch18 LR `0.05` benefit is causally real in the tested setup and survives removal of inherited S17 momentum. Inherited momentum measurably affects the secondary within-epoch online training-loss outcome, but modification of the primary validation-loss LR effect is not established. Inherited S17 momentum therefore cannot by itself explain the endpoint response; deterministic and stochastic finite-step mechanisms remain unresolved.

MOMENTUM-RESET MECHANISM STAGE CLOSED — NEXT MECHANISTIC QUESTION NOT YET DESIGNED
