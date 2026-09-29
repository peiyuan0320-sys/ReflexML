<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# ReflexML Mechanism Experiment: Immediate LR-Switch Counterfactual

**Protocol ID:** `ReflexML-Mechanism-LR-Switch-v1`  
**Status:** FROZEN  
**Relationship to Phase 5:** Post-Phase-5 mechanism experiment. Phase 5 design, execution, estimands, results, and frozen conclusions remain unchanged.  
**Execution authorization:** None implied by this document. Implementation and independent implementation audit must precede any production execution.

---

# 1. Scientific question

Phase 5 established a strong short-horizon Now-vs-Wait3 validation-loss difference at horizons \(h=1,2,3\), followed by an abrupt collapse at \(h=4\).

Subsequent read-only diagnostics established that:

1. the \(h=3\rightarrow4\) collapse is dominated by a sharp Wait3-side catch-up;
2. the catch-up appears simultaneously in training loss, validation loss, and validation accuracy;
3. independent B-block future replicas reproduce the collapse and subsequent near-overlap;
4. the Wait3 catch-up occurs at epoch 18, coincident with the Wait3 learning-rate transition from \(0.1\) to \(0.05\);
5. existing Phase 5 data contain no counterfactual in which the same historical Wait3 epoch17 state continues at LR \(0.1\) during epoch18.

Therefore the unresolved causal question is:

> **Does the epoch18 learning-rate transition from \(0.1\) to \(0.05\) causally contribute to the sharp Wait3 catch-up observed at epoch18?**

This experiment tests the immediate one-epoch causal effect of that LR transition.

It does not attempt to explain the full optimizer mechanism, parameter-space dynamics, or long-horizon behavior.

---

# 2. Experimental contrast

For each selected historical Wait3 epoch17 state,

\[
S_{17},
\]

construct two counterfactual epoch18 continuations from the same reconstructed state:

### Switch

\[
\eta_{18}=0.05
\]

which reproduces the historical Wait3 treatment.

### Continue

\[
\eta_{18}=0.10
\]

which removes the historical epoch18 LR transition.

All other scientifically relevant state and future-randomness components must be identical between the two branches.

Thus the intended treatment contrast is:

\[
\boxed{
\eta_{18}=0.05
\quad\text{vs.}\quad
\eta_{18}=0.10
}
\]

and the only intended treatment difference is the epoch18 global learning rate.

---

# 3. Experimental population and fixed sample

The experiment uses the same 36 outcome-independent Phase 5 primary states.

Freeze:

\[
\boxed{N=36}
\]

For every state, use exactly two pre-existing frozen A-block future streams:

\[
\boxed{\text{replica\_id}=1,2}
\]

Therefore:

\[
\boxed{K=2}
\]

and the experiment contains:

\[
36\times2=72
\]

state-replica units.

The 72 units are **not** 72 independent states.

The state remains the population-level analysis unit.

Replica selection is identity-based and must not depend on historical or new outcomes.

No substitution of A3 or another replica is allowed because an A1/A2 outcome is inconvenient, surprising, missing, or unfavorable.

If a selected historical identity cannot be reconstructed or validated, execution stops for investigation rather than changing the sampling rule.

---

# 4. Historical epoch17 state reconstruction

The primary implementation strategy is direct historical reconstruction.

For each selected (base_run_id, A, replica_id) unit, reconstruct the historical Wait3 epoch17 state from the corresponding frozen Phase 5 artifact.

The reconstructed training state must include:

all model parameters;

complete SGD per-parameter state required by the frozen optimizer;

all momentum buffers;

the optimizer parameter-group configuration;

epoch17 LR (=0.1);

epoch counter (=17);

the correct epoch18 DataLoader generator state;

frozen data/split/configuration identity.

Historical model parameters and momentum buffers stored as numpy_npy_base64_v1 must be reconstructed without numerical transformation.

The epoch17 artifact is authoritative for the epoch17 model tensors and momentum-buffer tensors.

The complete optimizer param_groups metadata is not stored in the per-epoch tensor evidence. It must therefore be reconstructed from the frozen epoch14 checkpoint together with the frozen production optimizer construction/configuration and training semantics.

For every selected unit, implementation must verify that these sources agree on:

optimizer class: torch.optim.SGD;

one parameter group;

parameter membership and ordering;

LR semantics;

momentum (=0.9);

weight decay (=0);

dampening (=0);

Nesterov False.

The parameter-to-momentum mapping must follow the frozen model's actual parameter identities and ordering. For the current model this mapping must be validated against the production model definition rather than inferred only from tensor shapes.

Any disagreement between the epoch14 checkpoint, frozen production configuration/code, model parameter identities, or reconstructed optimizer mapping is an integrity failure and must stop execution before Stage I.

For the current Phase 5 model and optimizer, no scheduler, BatchNorm state, model buffers, dropout state, or additional per-parameter SGD state has been identified as required.

Historical Python, NumPy, and torch global RNG states at epoch17 are not available in the A-block artifact. Under the frozen current setup, these RNGs do not affect the epoch18 SGD trajectory because the model has no stochastic layer, the data pipeline has no stochastic augmentation, execution is CPU with num_workers=0, and training shuffle is controlled by the independent DataLoader generator.

This justification is specific to the current frozen training system and must not be generalized to other models.

---

# 5. Epoch18 future-randomness reconstruction

For each selected unit, the frozen Phase 5 future seed is authoritative.

The historical epoch18 DataLoader generator state must be reconstructed using the same DataLoader construction and iteration semantics as the frozen production training path.

It is not sufficient to independently generate a nominal sampler permutation.

For each selected unit:

construct the training DataLoader using the frozen production data, subset, batch-size, shuffle, generator, worker, and loader semantics;

initialize its independent training generator from the frozen future seed exactly as in Phase 5;

create and exhaust the complete DataLoader iteration corresponding to epoch15;

repeat the same complete DataLoader iteration semantics for epochs16 and 17;

do not perform model forward passes, backward passes, optimizer steps, or parameter updates during this reconstruction;

record the realized sample-index order and batch boundaries yielded by each complete iteration;

after the epoch17 DataLoader iteration is exhausted, capture the resulting generator state as the reconstructed starting generator state for epoch18.

The reconstructed realized orders

\[
O_{15},O_{16},O_{17}
\]

must exactly match the corresponding historical wait_realized_orders and wait_epoch_order_sha256.

The reconstructed epoch18 loader must then yield

\[
O_{18}
\]

exactly matching the historical epoch18 realized order and hash.

The historical \(O_{18}\) validation iteration must be performed from an independent copy of the reconstructed post-epoch17 generator state; validation must not advance or mutate the canonical generator state subsequently cloned for Switch and Continue.

Thus the historical order checks validate the complete loader-iteration reconstruction, not merely a separately generated sampler permutation.

Both Switch and Continue must start from independent copies of the same reconstructed epoch18 DataLoader generator state and must therefore receive the same realized epoch18 minibatch sequence and boundaries.

No alternative fixed-order loader, manually supplied permutation, or simplified sampler implementation may silently replace the frozen production DataLoader semantics.

Failure of any order or boundary comparison is an integrity failure and stops the experiment before scientific interpretation.

---

# 6. Primary outcome

The primary outcome is:

\[
\boxed{\text{epoch18 validation loss}}
\]

For state \(s\) and replica \(r\), define:

\[
L^{S}_{sr}
\]

as epoch18 validation loss under Switch LR \(0.05\), and

\[
L^{C}_{sr}
\]

as epoch18 validation loss under Continue LR \(0.10\).

The historical Phase 5 Switch value remains the authoritative Switch outcome after successful reproduction.

The new Switch execution exists as an integrity reproduction gate, not as a new independent scientific observation.

---

# 7. Primary causal estimand

For each state-replica unit define:

\[
\delta_{sr}
=
L^{C}_{sr}-L^{S}_{sr}.
\]

The primary causal estimand is:

\[
\boxed{
\delta_{\mathrm{switch}}
=
E_{S,\omega}
\left[
L^{C}(18)-L^{S}(18)
\right]
}
\]

with sign convention:

\[
\delta_{\mathrm{switch}}>0
\]

meaning that switching to LR \(0.05\) produces lower epoch18 validation loss than continuing at LR \(0.10\).

The state-level estimate is:

\[
\bar{\delta}_s
=
\frac{\delta_{s1}+\delta_{s2}}{2}.
\]

The overall estimator is:

\[
\boxed{
\hat{\delta}_{\mathrm{switch}}
=
\frac{1}{36}
\sum_{s=1}^{36}\bar{\delta}_s
}
\]

Replicas are therefore averaged within state before population-level inference.

---

# 8. Catch-up decomposition

For each unit define the historical Switch catch-up:

\[
D_{sr}
=
L_{17,sr}-L^{S}_{sr},
\]

and the counterfactual Continue catch-up:

\[
\kappa^{C}_{sr}
=
L_{17,sr}-L^{C}_{sr}.
\]

By construction:

\[
\boxed{
D_{sr}
=
\kappa^{C}_{sr}
+
\delta_{sr}
}
\]

exactly, up to floating-point representation.

At the aggregate level:

\[
\boxed{
\hat D
=
\hat\kappa_C
+
\hat\delta_{\mathrm{switch}}
}
\]

must also hold numerically.

This decomposition separates:

- catch-up occurring even if LR remains \(0.1\);
- additional epoch18 effect attributable to switching LR to \(0.05\).

The descriptive fraction

\[
R=
\frac{\hat\delta_{\mathrm{switch}}}{\hat D}
\]

may be reported if \(\hat D\neq0\).

\(R\) is descriptive only.

It must not be truncated to \([0,1]\), and no threshold such as 25%, 50%, or 75% may be introduced post hoc to classify a mechanism as dominant.

---

# 9. Secondary outcomes

Pre-specified secondary outcomes are:

1. epoch18 training loss;
2. epoch18 validation accuracy.

For each outcome, use the same paired Switch-vs-Continue construction.

These outcomes assess whether the immediate counterfactual effect is consistent across metrics.

They do not replace validation loss as the primary outcome.

No composite outcome will be constructed.

No secondary outcome may be promoted to primary because its result is stronger or more favorable.

---

# 10. Uncertainty estimation

Population uncertainty is evaluated at the **state level**.

For the primary estimand:

1. compute the two-replica mean \(\bar\delta_s\) for every state;
2. resample the 36 states with replacement;
3. when a state is sampled, retain its complete two-replica unit;
4. calculate the overall mean for each bootstrap draw.

Use:

\[
\boxed{B=10{,}000}
\]

bootstrap replicates.

RNG implementation:

`numpy.random.Generator(PCG64)`

Frozen bootstrap seed:

\[
\boxed{12114954806208443255}
\]

derived as the unsigned integer represented by the first 16 hexadecimal characters of:

`SHA256("ReflexML-Mechanism-LR-Switch-v1-bootstrap")`

whose complete digest is:

`a820f7355df733771508e678b4cadc0a191cac0f089911879390118ec078e069`

Report the percentile:

\[
[2.5\%,97.5\%]
\]

interval.

The same state-level bootstrap construction may be used for the pre-specified secondary quantities \(D\) and \(\kappa_C\) and for secondary outcomes.

No EIV correction, latent-variable model, replica-level pseudo-independent bootstrap, or alternative bootstrap procedure belongs to the primary analysis.

---

# 11. Stage I — historical Switch reproduction gate

No Continue counterfactual may be executed for scientific analysis until the complete historical Switch reproduction gate passes.

For all:

\[
36\times2=72
\]

selected units:

1. reconstruct the historical epoch17 state;
2. reconstruct the historical epoch18 DataLoader state;
3. set epoch18 LR to the historical value \(0.05\);
4. execute exactly one epoch using the production-equivalent training semantics;
5. compare the result against the corresponding frozen historical Wait3 epoch18 artifact.

For every unit, the following must match historical evidence:

- LR;
- complete realized minibatch order and boundaries;
- epoch-order SHA-256;
- training loss;
- validation loss;
- validation accuracy;
- all final model tensors;
- all final SGD momentum buffers.

Model and momentum tensors must match exactly in dtype, shape, and canonical numerical bytes.

Under the same approved numerical environment, metric values are also required to match exactly.

No tolerance may be introduced after observing a mismatch.

The gate criterion is:

\[
\boxed{72/72\text{ PASS}}
\]

Anything less than 72/72 is:

\[
\boxed{\text{GATE FAIL}}
\]

and stops the experiment before Continue execution or interpretation.

A Switch reproduction failure is a technical/integrity failure, not a scientific result.

It must never be interpreted as evidence for or against an LR-switch mechanism.

---

# 12. Stage II — Continue counterfactual

Stage II is permitted only after Stage I has formally passed 72/72.

For every selected unit:

1. reconstruct the same immutable historical epoch17 state;
2. restore the same reconstructed epoch18 DataLoader generator state;
3. retain LR \(0.10\);
4. execute exactly one epoch;
5. record the same outcome and integrity fields as for Switch.

The Continue branch must not inherit any mutable state advanced by the Switch reproduction run.

Switch and Continue must be instantiated independently from the same frozen epoch17 source state.

The intended treatment difference must remain:

\[
\boxed{\text{epoch18 LR only}}
\]

---

# 13. Execution order and outcome isolation

Execution order is frozen as:

\[
\boxed{
\text{Stage I: all 72 Switch reproductions}
\rightarrow
\text{formal 72/72 gate}
\rightarrow
\text{Stage II: 72 Continue branches}
}
\]

Continue outcomes must not be generated before the complete Stage I gate has passed.

Implementation testing before production execution may use synthetic/toy inputs only.

Synthetic tests are not evidence that the production reproduction gate has passed.

---

# 14. Technical failures and retries

A scientifically valid but surprising outcome is never a reason to retry.

A production unit may be rerun only after a documented technical failure that prevented production of a valid scientific outcome.

Any retry must preserve:

- state identity;
- replica identity;
- historical epoch17 source;
- future seed;
- treatment assignment;
- scientific code semantics.

Technical failures and retries must remain visible in the execution record.

They must not be silently deleted or replaced.

If a Stage I unit completes but fails historical reproduction, this is an integrity-gate failure, not an ordinary stochastic retry condition.

The experiment must stop and the cause must be investigated.

If correction requires changing scientific semantics, reconstruction semantics, treatment semantics, or analysis rules, a new protocol version must be frozen before execution resumes.

---

# 15. Primary interpretation rules

The experiment is intended to distinguish causal contributions, not force a binary mechanism verdict.

### Switch effect

Recall the sign convention:

\[
\delta_{\mathrm{switch}}
=
E[L^C(18)-L^S(18)].
\]

Therefore positive values favor Switch and negative values favor Continue.

If the pre-specified 95% state-level bootstrap interval for

\[
\delta_{\mathrm{switch}}
\]

lies entirely above zero, the experiment supports an immediate causal reduction in epoch18 validation loss from switching LR \(0.1\to0.05\) under this experimental setting.

If the interval lies entirely below zero, the experiment supports the opposite directional result:

> under this experimental setting, switching from LR \(0.1\) to \(0.05\) produces higher epoch18 validation loss than continuing at LR \(0.1\).

This negative-direction result must be reported as observed and must not be reclassified as technical failure merely because it contradicts the motivating mechanism hypothesis.

If the interval includes zero, the direction of the immediate Switch effect is not resolved by the pre-specified uncertainty procedure.

An interval containing zero must not be described as proof of no effect, proof of equivalence, or evidence that the two learning rates are identical.

No equivalence margin is specified in this protocol.

All three possible primary outcomes—

\[
CI>0,\qquad CI<0,\qquad 0\in CI
\]

—are scientifically admissible results and do not authorize alteration of the frozen sampling, estimand, analysis, or interpretation rules.

### Continue catch-up

If the pre-specified 95% state-level bootstrap interval for

\[
\kappa_C
\]

lies entirely above zero, the experiment supports the conclusion that the Continue-\(0.1\) trajectory itself exhibits epoch17→18 validation-loss catch-up.

If the interval lies entirely below zero, the Continue-\(0.1\) trajectory shows movement in the opposite direction over epoch17→18 under the pre-specified sign convention.

If the interval includes zero, the direction of Continue catch-up is not resolved by the pre-specified uncertainty procedure.

An interval containing zero must not be interpreted as proof that Continue exhibits no change.

### Joint interpretation

If both \(\delta_{\mathrm{switch}}\) and \(\kappa_C\) show supported positive contributions, the data support contributions from both endogenous trajectory evolution and the LR transition.

If Continue catch-up is supported while an additional Switch effect is not established, the simple claim that the LR transition is necessary for the immediate catch-up is weakened.

If Switch effect is supported while Continue catch-up is not established, the evidence is more consistent with the LR transition contributing substantially to the immediate catch-up.

If uncertainty prevents either contrast from being resolved, the result must be reported as insufficiently precise rather than repaired by increasing \(K\) after outcome inspection.

If either contrast has a supported direction opposite to the motivating expectation, that result must be reported directly and must not be reclassified, discarded, or repaired by changing the frozen design.

No categorical “dominant mechanism” threshold is pre-specified.

The numerical decomposition must be reported directly.

---

# 16. Prohibition on adaptive sample expansion

The frozen design is:

\[
\boxed{N=36,\ K=2}
\]

No additional replicas may be added after Continue outcomes are inspected in order to obtain a narrower interval, desired sign, desired significance result, or cleaner mechanism narrative.

If the frozen experiment is genuinely insufficiently precise, that is the result of this experiment.

Any subsequent increase in \(K\) requires a separately motivated, prospectively frozen follow-up experiment.

---

# 17. Non-claims

This experiment does **not** by itself establish:

- parameter-space reconvergence;
- function-space reconvergence;
- optimizer-state reconvergence;
- Hessian or sharpness dynamics;
- a general SGD mechanism;
- a general neural-network training law;
- long-horizon necessity or sufficiency of the LR switch;
- external generality across architectures, datasets, optimizers, or schedules;
- novelty relative to the broader learning-rate-transition literature.

Loss-space overlap must not be renamed parameter or functional reconvergence without direct measurements.

The experiment concerns the immediate epoch18 causal contrast in the frozen ReflexML training system.

---

# 18. Explicitly out of scope

This protocol does not authorize:

- Wait1;
- Wait2;
- Wait4;
- Never-switch trajectories beyond the single Continue epoch18 counterfactual;
- additional horizons after epoch18;
- parameter-distance analysis;
- representation similarity;
- Hessian analysis;
- sharpness analysis;
- optimizer-state perturbation experiments;
- additional \(K\);
- external replication;
- new scheduler experiments;
- infrastructure/control-plane redesign.

Such experiments may be considered only after this experiment is completed and interpreted.

---

# 19. Relationship to Phase 5

This is a new post-Phase-5 mechanism experiment.

It does not modify:

- Phase 5A estimands;
- Phase 5B estimands;
- Phase 5 execution records;
- Phase 5 frozen conclusions;
- EIV sensitivity results;
- precision-planning results.

The Phase 5 result:

**“5A signature supported.”**

remains unchanged.

The Phase 5B result:

**“5B primary association insufficiently precise.”**

remains unchanged.

This mechanism experiment addresses a different causal question.

---

# 20. Implementation boundary

Scientific design is controlled by this protocol.

Implementation may determine only the minimal software mechanics necessary to execute the frozen design faithfully.

Implementation must not independently change:

- scientific question;
- treatment contrast;
- sample identities;
- \(N\);
- \(K\);
- primary outcome;
- estimand;
- sign convention;
- bootstrap procedure;
- gate requirements;
- interpretation rules;
- stopping rules.

The preferred implementation is one small isolated mechanism-test runner reusing existing frozen training semantics.

No new orchestration platform, scheduler, control plane, or generalized experiment framework is scientifically required.

---

# 21. Required pre-execution sequence

The required workflow is:

\[
\boxed{
\text{Protocol freeze}
\rightarrow
\text{Implementation}
\rightarrow
\text{Independent implementation audit}
\rightarrow
\text{Stage I Switch gate}
\rightarrow
\text{Stage II Continue}
\rightarrow
\text{Frozen analysis}
}
\]

Before production execution, the final protocol bytes and SHA-256 must be recorded.

Implementation audit must be independent of the implementation task.

The implementation audit must verify production behavior against this protocol rather than redesign the scientific experiment.

No real Continue outcome may be generated before:

1. protocol freeze;
2. implementation completion;
3. independent implementation approval;
4. complete Stage I 72/72 Switch reproduction PASS.

---

# 22. Frozen decision boundary

Once this protocol is formally frozen, any change to a scientific field requires a new version.

Implementation bugs may be repaired without changing the scientific design only if the repair restores compliance with the already-frozen protocol.

Scientific changes must not be disguised as implementation repairs.

Unexpected results do not authorize protocol modification.

The experiment ends after the pre-specified Stage II outcomes and frozen analysis are complete.

Any next mechanism question requires a new prospective design.

---

# 23. Core experiment summary

The experiment can be reduced to:

\[
\boxed{
S_{17}
\rightarrow
\begin{cases}
LR=0.05 & \text{historical Switch}\\
LR=0.10 & \text{counterfactual Continue}
\end{cases}
}
\]

with identical historical epoch18 future randomness.

Primary causal estimand:

\[
\boxed{
\delta_{\mathrm{switch}}
=
E[L_C(18)-L_S(18)]
}
\]

Catch-up decomposition:

\[
\boxed{
D=\kappa_C+\delta_{\mathrm{switch}}
}
\]

Frozen sample:

\[
\boxed{
N=36,\quad K=2,\quad A1/A2
}
\]

Integrity requirement:

\[
\boxed{
72/72\text{ exact historical Switch reproductions before Continue}
}
\]

The purpose is not to find a story that explains the Phase 5 result.

The purpose is to perform the smallest controlled counterfactual capable of separating immediate LR-switch causation from catch-up that would have occurred under continued LR \(0.1\).
