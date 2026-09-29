<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# ReflexML LR-Switch Dynamics v1

Protocol ID: `ReflexML-LR-Switch-Dynamics-v1`.

Question: when does the already established LR-switch causal validation-loss
contrast form within epoch18? This localizes timing; it does not identify why.
This is a separately specified A1 temporal follow-up, not a revision of the
historical A1/A2 endpoint protocol or its scientific results.

Frozen before any new replay or interior evaluation:

- Exactly base_run_id 1–36, historical Phase5A replica A1. No A2–A15.
- Independent reconstruction from the same historical Wait3 S17 model and
  complete SGD optimizer state, including inherited momentum, for both arms.
- Switch LR .05; Continue LR .10. All other configuration and future minibatch
  order/boundaries identical to historical LR-Switch. Exactly 79 updates.
- Probes exactly `{0,1,3,8,16,32,48,64,78,79}`. t0 precedes update 1; tk follows
  update k. Capture complete model state_dict through the existing training
  callback and Collapse Dynamics snapshot primitive. No validation in training.
- Offline objective: historical CrossEntropyLoss and sample-weighted evaluation
  on the same full 5,000 validation examples. No test-set use.
- d_t(s) = L_10,t(s) - L_05,t(s); D_t = mean_s d_t(s); D_0 = 0.
  Preserve both M_05,t and M_10,t. G_t = D_t / D_79 is descriptive,
  untruncated, and undefined if D_79 = 0. No threshold is an execution gate.
- Paired whole-state bootstrap, NumPy Generator(PCG64), seed
  `10698172564239836591`, B=20000, percentile pointwise 95% intervals.
  Resample 36 paired state trajectories; never steps or independent arms.
  A1 estimates do not average over future randomness. Sparse probes cannot
  exclude between-probe excursions. No simultaneous-band claim.

## Historical authority and exactness

Reuse the Phase5 ledger, checkpoint/branch identity, future mapping, S17 tensor
reconstruction and epoch18 sampler reconstruction in existing modules.
Switch endpoints: `internal-artifacts/stage_i_gate.json`,
SHA256 `dd7d4273a215f1c71cdf816bce3334dc846c37e31b92f0e29e46b7748d67dc80`.
Continue endpoints: `internal-artifacts/stage_ii_completion_manifest.json`,
SHA256 `b4db6b50f176c556ec6ec31182831bbe2a308f174dca499bb44e1503c0e5d473`.
Read only the enumerated A1 record for each state/arm, verify its exact hash,
source/checkpoint/future identities, order, epoch and LR. Switch also reuses
`check_switch`; Continue receives the same exact scalar/tensor equality standard.
No tolerances. The gate checks S17 model and optimizer, all endpoint model and
momentum tensors, optimizer metadata, LR, train loss, validation loss/accuracy,
order and 79 updates. Gate/read/resume binds every snapshot hash, implementation
hash and runtime identity. Any mismatch hard-stops; no automatic retry.

Recoverability labels: A = stored interior trajectory; B = sources/order/endpoints
available for exact replay, pending replay gate; C = missing/conflicting source.
This follow-up is Level B until operator-run instrumented endpoint gates PASS.
Historical Continue completion is endpoint evidence, not an instrumented
Continue reconstruction PASS.

## Execution stages and boundaries

Operator runs preflight state 1 A1 Switch and Continue in a separate root and
runs endpoint gates; no report or interior interpretation. Production launcher
requires both preflight PASSs under the current code/runtime. It runs states
1–36 sequentially: Switch replay -> gate -> Continue replay -> gate, 72 replays.
It never reads interior losses or launches reports. A complete identity-bound
PASS may be skipped on resume; incomplete/failed attempts, changed hashes,
changed runtime or ambiguous outputs stop for investigation. Never overwrite
or delete prior attempts. Dry-run writes nothing and clearly reports missing
preflight gates. Preflight data are not pooled into production.

Codex is authorized for inspection, implementation, unit/static/dry-run/source
reconstruction checks only. The user runs preflight and production. Analysis
is a separate explicit command after complete 72/72 exact gates; it is not
launched in this implementation handoff.

## Interpretation after separate analysis

Report primary table t, M_05,t, M_10,t, D_t, 95% CI, and descriptive G_t.
Compare G_8/G_16/G_32 with historical A1 collapse closure approximately
.56/.775/.936 (user-supplied reference; link the frozen Collapse Dynamics source
before final scientific reporting). Discuss immediate, early accumulation,
gradual, late or non-monotone shape without inventing outcome-based cutoffs.
Relative timing similarity does not prove LR is the only mechanism; difference
does not negate the established endpoint causal effect. Attribute arm movements
descriptively. Preserve reversals/overshoots and unresolved results.

No gradient noise, D/N decomposition, optimizer/gradient/update/momentum/cosine
norms, clustering, m4/m8, 80-point validation, general SGD study, or automatic
next experiment. Stop after implementation and commands in this handoff.
