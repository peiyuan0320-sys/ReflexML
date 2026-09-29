<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# ReflexML Momentum Reset: frozen scientific protocol

**Protocol ID:** `ReflexML-Momentum-Reset-v1`

**Status:** `FROZEN`

**Execution authorization:** `execution_authorized = false`

**Scope:** Prospective, post-Phase-5 mechanism experiment. This document is frozen; its bytes and SHA-256 are recorded in `MOMENTUM_RESET_PROTOCOL_FREEZE_RECORD.md`. It does not change the frozen Phase 5 or LR-Switch protocols, evidence, results, or conclusions.

## 1. Scientific question and boundary

The completed LR-Switch experiment found an immediate epoch18 validation-loss benefit from reducing the learning rate (LR) from 0.10 to 0.05 from the historical Wait3 epoch17 state. Its frozen primary contrast, `Continue - Switch`, is `0.041100323`, with a 95% state-level bootstrap interval of `[0.033847031, 0.048712969]`. Prespecified epoch18 online training loss and validation accuracy effects were directionally consistent. The dynamical mechanism remains unknown.

The next question is: **Does the inherited epoch17 SGD momentum state causally modify the immediate epoch18 LR effect?** This experiment tests one joint optimizer-state intervention over one epoch. It is not designed to identify a complete optimization mechanism.

## 2. Factorial intervention

Use the existing historical Wait3 epoch17 states. The two factors are inherited momentum-buffer values and epoch18 LR:

| Inherited momentum | LR 0.10 | LR 0.05 |
| --- | --- | --- |
| Keep historical values | Existing Continue | Existing Switch |
| Reset values to zero | New Reset-Continue | New Reset-Switch |

The Keep cells already exist and must not be rerun merely for this experiment. Only the two Reset cells require new scientific training.

For each state-replica unit, reconstruct the exact historical S17 model parameters and the complete historical SGD optimizer structure. Preserve every existing historical momentum-buffer entry and its parameter association; replace only its numerical value with an exact all-zero tensor matching the associated historical buffer in shape, dtype, and device. Preserve parameter and optimizer-state identity and mapping. Reset **all four** historical momentum buffers jointly. Preserve momentum coefficient `0.9`, dampening `0`, weight decay `0`, Nesterov `False`, and optimizer parameter-group membership and order. Set epoch18 LR to the cell's assigned value, then run exactly one epoch18 on the paired canonical future stream. After the reset, momentum SGD operates normally and newly accumulates momentum during epoch18.

The intervention is `do(inherited S17 momentum-buffer values = 0)`. It is neither `momentum coefficient = 0` nor deletion of momentum-buffer entries. No other optimizer state or training variable may be changed as part of Reset.

## 3. Fixed population, reconstruction, and pairing

Use exactly `N = 36` independent historical states and exactly `K = 2` future replicas per state: the historical A1 and A2 identities. There are 72 state-replica units and exactly `36 × 2 × 2 = 144` new one-epoch Reset branches. Epoch15–17 model training replay is not required. There are no outcome-dependent exclusions, additional replicas, adaptive increases in K, or substitutions of states or replicas.

Within each state-replica unit, all four cells must correspond to the same S17 model parameters, data and split, state and replica identities, future seed, epoch18 sample order and batch boundaries, training-step count, evaluation data, model architecture, and optimizer hyperparameters other than the two intended factors. Keep and Reset differ only in inherited momentum-buffer numerical values. Within a momentum condition, LR 0.10 and LR 0.05 differ only in LR. Verify realized order and boundaries; a shared seed alone is insufficient evidence of pairing.

Reconstruct each Reset branch independently from the immutable historical S17 source. Do not obtain one Reset branch by training or mutating the other. Validate the historical model, four buffers, complete optimizer parameter groups, and canonical post-S17 DataLoader-generator state before applying the Reset intervention. If required source identity or pairing cannot be established, stop rather than substitute a unit.

## 4. Authority of the existing Keep cells and compatibility gate

The top row is immutably bound to the completed LR-Switch experiment. The frozen historical primary-analysis authority is `internal-artifacts/primary_analysis.json` (SHA-256 `9b7d8a940c3b62590cad1793070c523ddb0c06c8592d2abe4d7c84378454e791`). `L_05K` comes only from the historical Switch A1/A2 records enumerated, by exact filename and record SHA-256, in `internal-artifacts/stage_i_gate.json` (SHA-256 `dd7d4273a215f1c71cdf816bce3334dc846c37e31b92f0e29e46b7748d67dc80`; Stage I reproduction gate `72/72 PASS`). `L_10K` comes only from the historical Continue A1/A2 records enumerated, by exact filename and record SHA-256, in `internal-artifacts/stage_ii_completion_manifest.json` (SHA-256 `b4db6b50f176c556ec6ec31182831bbe2a308f174dca499bb44e1503c0e5d473`; Stage II `72/72 COMPLETE — INTEGRITY VERIFIED`). Resolve each enumerated filename relative to its bound Stage I or Stage II directory. The bound record selectors are exactly `switch-{base_run_id:03d}-A-r{replica_id:02d}.json` and `continue-{base_run_id:03d}-A-r{replica_id:02d}.json`, for `base_run_id` 1–36 and `replica_id` 1–2 (historical A1/A2), respectively. Require the record bytes to match the SHA-256 listed for that filename in the respective bound gate or manifest; require exact matching state-replica identities across the two sets. These are the same 36 historical states and exactly A1/A2, with no choice among alternative historical records. No substitution, recomputation, rerun, cleaner copy, later-analysis output, or regenerated Keep cell may replace the bound records. Reuse the Keep observations read-only. Before authorizing any Reset scientific training, an independent implementation audit must establish that Reset execution uses the same relevant numerical training semantics and a compatible environment as these Keep cells.

The production Stage II mechanism implementation SHA-256 was `f05c0d3502f09e12f24ccb43ad39b092768a985748b5f684db58dd034f34ace0`; the later analysis-patched implementation SHA-256 is `d863c8eb5db64d6e301360880388aea1d898c833f34a24ee826fd60287c7b3e2`. The audit must inspect and explain their difference and determine whether it affects the numerical training path. A matching label or passing test alone does not establish compatibility. If numerical training compatibility cannot be established, Reset execution is **BLOCKED** until resolved. Do not silently rerun or replace the historical Keep cells, and do not redesign infrastructure to evade this gate.

## 5. Primary outcome and factorial estimands

The sole primary outcome is **epoch18 validation loss**. It cannot be replaced or superseded after execution. For each state-replica unit, let `L_10K`, `L_05K`, `L_10R`, and `L_05R` be validation loss in the respective LR and momentum cells. Compute:

\[
\Delta_K=L_{10K}-L_{05K},\qquad
\Delta_R=L_{10R}-L_{05R},\qquad
I=\Delta_K-\Delta_R.
\]

Positive `Delta_K` or `Delta_R` means LR 0.05 has lower validation loss in that momentum condition. The **primary estimand is the factorial interaction** `I`. Also compute the Reset effects at fixed LR:

\[
R_{.10}=L_{10K}-L_{10R},\qquad
R_{.05}=L_{05K}-L_{05R},\qquad
I=R_{.10}-R_{.05}.
\]

Positive `R_eta` means Reset lowers validation loss at that LR. Verify the algebraic identity at the unit and aggregate levels, allowing only declared floating-point representation differences. Always report `I`, `Delta_K`, `Delta_R`, `R_.10`, and `R_.05`; never interpret the interaction without the simple effects.

## 6. Keep-row integrity check

Before the new factorial analysis, reconstruct `Delta_K` from the **bound existing A1/A2 Keep cells** in Section 4 and check it against the bound historical primary-analysis artifact. Under the original frozen LR-Switch numerical rules and state-level bootstrap (including its historical seed), reproduce the frozen `Continue - Switch` primary estimate at full stored precision, `0.04110032269702188` (reported as `0.041100323`), and its historical 95% interval `[0.033847031257023194, 0.04871296943619432]` (reported as `[0.033847031, 0.048712969]`). This is an **integrity check of the historical frozen result**, not a new Momentum-Reset uncertainty calculation. The new factorial bootstrap in Section 7 uses its own frozen seed and state-resampling procedure; its `Delta_K` interval is reported separately and need not reproduce the historical LR-Switch interval endpoints. The historical Keep point estimate and underlying bound cell data must remain consistent with the frozen historical authority. Any mismatch in bound source identity, record hash, pairing, or historical reconstruction blocks scientific interpretation; do not repair it by choosing another Keep source or modifying the frozen LR-Switch artifact.

## 7. Aggregation and primary uncertainty

The independent population unit is **state**, not replica. For each estimand, calculate the paired contrast separately in A1 and A2, average the two contrasts within each state, and then average across all 36 states. The 72 replicas must not be treated as independent population units.

Use a state-level percentile bootstrap with `B = 10,000`. Resample the 36 states with replacement, retaining the complete paired A1/A2 structure and all four factorial cells for every sampled state occurrence. Use NumPy `Generator(PCG64)` with seed `6630267477959695398`, the unsigned integer represented by the first 16 hexadecimal digits of SHA-256 of `ReflexML-Momentum-Reset-v1-bootstrap`. The full digest is `5c0371247d9cdc26aec35a6cf9b400aa31813d58a78db4230000230b470a1bc1`. Report percentile `q025` and `q975` for each quantity. Reuse the **same state-resampling indices** for `I` and every simple effect, including the secondary analyses.

There is no replica-level pseudo-independent bootstrap, adaptive alternative uncertainty method, or post-hoc p-value requirement.

## 8. Prespecified primary interpretation

- If the 95% interval for `I` lies entirely above zero, evidence supports that resetting inherited momentum **attenuates** the LR 0.05 validation-loss advantage relative to Keep. This establishes causal effect modification by inherited momentum state in this setup, without identifying why it occurs.
- If the interval lies entirely below zero, evidence supports that resetting inherited momentum **amplifies** that advantage. This establishes effect modification in the opposite direction.
- If the interval includes zero, inherited-momentum modification of the LR effect is **not established with this experiment**. This is neither proof of no interaction nor equivalence. No equivalence margin is prespecified, and none may be invented after results are seen.

Interpret `I` together with `Delta_R`, `R_.10`, and `R_.05`. If Reset attenuates the Keep-row LR advantage, distinguish whether Reset improves LR 0.10 more than LR 0.05, harms LR 0.05 more than LR 0.10, or moves both cells in ways that yield the interaction. Do not compress these different patterns into “bad momentum.” Predominant improvement at LR 0.10 is more consistent with an inherited-state contribution particularly unfavorable under high LR. Predominant worsening at LR 0.05 weakens the simple claim that high LR was merely harmed by bad inherited momentum; a different LR × inherited-state interaction is present. If `Delta_R` remains clearly positive, the immediate LR advantage persists after removing inherited S17 momentum, so inherited S17 momentum cannot by itself explain the LR effect. If `Delta_R` is unresolved, say so; an interval containing zero does not imply equivalence.

## 9. Prespecified secondary outcomes

Apply the same four-cell factorial arithmetic, state-first aggregation, and **same frozen bootstrap state indices** to (1) epoch18 online training loss and (2) epoch18 validation accuracy. For each, report `Delta_K`, `Delta_R`, `I`, `R_.10`, and `R_.05`. These analyses are secondary and cannot alter the primary validation-loss conclusion. There is no composite statistic, joint significance test, or promotion of a secondary outcome to primary.

Epoch18 online training loss is the existing implementation's sample-weighted mean of minibatch losses encountered along the epoch18 training trajectory; it is not automatically an endpoint full-training-set loss. Lower training loss is better. For validation accuracy, retain the **raw factorial arithmetic** above, including `Continue - Switch` in the Keep row. Do not reverse signs after observing results merely to make positive mean “better”; interpretation of accuracy must respect that larger accuracy is better.

## 10. Pretraining evidence and completion

Every Reset branch must preserve sufficient **pretraining** evidence to prove the intended intervention. At minimum, record:

- protocol ID and, after formal freeze, protocol SHA-256; implementation and environment identities;
- state ID, replica ID, historical S17 artifact and checkpoint/source identities, and future seed;
- S17 model-parameter identities and hashes; parameter-name ↔ optimizer-ID mapping;
- original historical momentum-buffer identities and hashes;
- zero-reset buffer identities, shapes, dtypes, and devices, plus a numerical all-zero check **before training** for every required buffer;
- complete optimizer-group settings and assigned LR;
- canonical post-S17 DataLoader-generator identity;
- realized epoch18 sample order and batch boundaries, training-step count, and epoch-order hash;
- epoch18 online train loss, validation loss, and validation accuracy;
- final model-tensor and momentum-buffer evidence; and
- branch-record hash.

The complete-set manifest must account for exactly 144 expected Reset branches, identify the first scientifically valid attempt or unresolved disposition for each, and link any technical-invalid attempt to its same-identity retry under Section 11. No missing, duplicate, substituted, or outcome-selected scientific unit is admissible. Use the smallest isolated extension compatible with the existing evidence pattern; do not build a new control plane or orchestration platform.

## 11. Technical failures and execution sequence

Classify every attempt using the following frozen, outcome-independent rules:

- **Technical-invalid attempt:** An attempt failed to produce a scientifically valid execution because of an objective non-scientific execution, infrastructure, or artifact failure, or a violation of required scientific identity, source state, RNG/future stream, treatment, environment, numerical-kernel compatibility, or integrity conditions. Its classification must be supported by evidence of the failed condition and must not depend on whether a training or validation metric is favorable, unfavorable, large, small, or finite. A retry is permitted only after this classification, and only for that branch with exactly the same state, replica, treatment, S17 source, model parameters, intended reset state, LR, canonical future RNG/minibatch stream, and scientific protocol. If the required identity cannot be restored or established, stop rather than retry by substitution.
- **Scientifically valid attempt:** The frozen scientific computation followed the required treatment, source state, future randomness, numerical kernel/environment, and execution semantics. Its numerical behavior is a substantive outcome even if extreme, divergent-looking, non-finite, NaN, Inf, or overflow-like, or if it prevents the usual finite contrast. Such behavior is not automatically a retryable machinery failure. Distinguish an objectively demonstrated machinery failure from numerical behavior generated by valid treatment execution. The **first scientifically valid attempt** is authoritative; it must never be rerun because its outcome is surprising, poor, inconvenient, or inconsistent with expectations. No best-run or outcome-based selection is permitted.

Preserve every attempt, including technical-invalid attempts, in an immutable history recording the attempt identifier, branch scientific identity, invalidity reason if applicable, failure stage, relevant integrity evidence, whether any scientific outcome was generated, and a link to any same-identity retry. Preserve valid numerical outcomes and their disposition even when they are non-finite; never recast substantive divergence as technical failure to rescue analysis.

If any required primary factorial cell remains substantively non-evaluable as a finite epoch18 validation-loss value after valid execution, the full prespecified primary factorial estimate and confidence intervals are **not evaluable under the frozen analysis as specified**. Preserve and report the complete 144-branch disposition, identifying each affected state, replica, and treatment, and whether its event was technical-invalid or substantive numerical behavior. Do not issue the normal `I`, `Delta_R`, `R_.10`, or `R_.05` causal interpretation as though all factorial cells were observed; do not create an authoritative available-case primary analysis. The primary Momentum-Reset scientific conclusion is **unresolved / non-evaluable under the frozen estimand**. Do not drop an affected state or replica, silently reduce `N` or `K`, impute after outcomes, replace a seed or replica, rerun until finite, or change the estimand. Scientifically valid execution may be complete even when the finite-valued estimand is non-evaluable. Any different handling requires explicit scientific change control through a future prospective protocol amendment independently approved before any replacement execution; it cannot be chosen after observing the affected treatment result.

The required order is:

1. formal scientific protocol freeze;
2. implementation;
3. independent implementation audit;
4. numerical-kernel and environment compatibility gate against the existing Keep cells;
5. authorized Reset execution;
6. integrity completion check;
7. frozen primary analysis;
8. secondary analysis.

No Reset scientific branch may execute before the independent audit and compatibility gate pass. A protocol freeze or gate pass does not itself grant execution authorization. **Current state: `execution_authorized = false`.**

## 12. Nonclaims and excluded work

Resetting all four buffers is an artificial joint state intervention. A causal Reset effect does not mean such a reset occurs naturally during ordinary training. This experiment does not identify gradient–momentum misalignment, momentum magnitude as the unique mechanism, an individual responsible buffer, overshoot, Hessian or sharpness mechanisms, deterministic finite-step dynamics, stochastic-gradient noise mechanisms, a natural mediation percentage, parameter or functional reconvergence, long-horizon necessity or sufficiency, external generality, a general SGD law, or novelty.

This experiment excludes momentum-coefficient sweeps; LR grids beyond 0.10 and 0.05; batch-size sweeps; Wait1, Wait2, or Wait4; more replicas or states; longer-horizon Reset branches; Hessian or sharpness scans; parameter-distance, representation-similarity, gradient/momentum alignment, or full-gradient virtual-update studies; and external model or dataset replication. Any such work requires a separate prospective question if later justified.

## 13. Version rule and frozen status

After formal freeze, a correction to the scientific question, intervention, factorial cells, sample identities, primary outcome or estimand, aggregation, uncertainty procedure, or interpretation rules requires a **new protocol version**, never a silent amendment. Purely mechanical implementation details that preserve scientific semantics may be resolved in implementation review and must remain auditable.

**This document is `FROZEN`; `execution_authorized = false`. No implementation, Reset training, or Reset-outcome inspection is authorized by this protocol.**
