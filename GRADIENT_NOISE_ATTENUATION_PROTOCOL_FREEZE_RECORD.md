<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# ReflexML Gradient-Noise Attenuation v1 — formal protocol freeze record

## Protocol identity

- Protocol ID: `ReflexML-Gradient-Noise-Attenuation-v1`.
- Protocol path: `internal-artifacts/GRADIENT_NOISE_ATTENUATION_PROTOCOL.md`.
- Freeze state: `FROZEN`.
- Frozen protocol SHA-256: `14561bb89471edbbd76b7b528848468f73603db14c72bf758db3113e5f4a99f6`.
- Frozen protocol byte size: `24947`.
- Pre-freeze audited candidate SHA-256: `8d41c5525876e08c5d45b39056dd85f7a5be20440ab5807ff822220091d93cb2` (`24935` bytes).

The frozen SHA above, not the pre-freeze draft SHA, is the scientific protocol identity for later implementation and execution work. The pre-freeze SHA is retained solely for audit provenance. The freeze changed only status and freeze metadata; scientific choices and interpretation rules remain those of the audited candidate.

## Audit history

1. **Initial independent audit:** P0 = 0, P1 = 1, P2 = 1; freeze blocked.
2. **Narrow repair:** fixed the canonical ordered `T` / local-position→global-example mapping and narrowed interpretation wording. No scientific design choice was changed.
3. **Targeted independent re-audit:** P0 = 0, P1 = 0, P2 = 0. Verdict supplied for this freeze: `PROTOCOL TARGETED RE-AUDIT PASS — READY TO FREEZE`.

## Frozen core design

- Population: 36 independent historical S17 states; selected replica A1 only.
- Reference: existing ordinary m=1 Momentum-Reset A1 cells, admitted by their bound records; no m=1 rerun.
- New intervention: m=4 same-parameter arithmetic mean of four component minibatch gradients, then exactly one momentum-SGD optimizer step at each of 79 positions. All four gradients at a position are evaluated at identical pre-update model parameters.
- LR treatments: `0.10` and `0.05`, paired within each state using identical S17 Reset state and four streams.
- Outcomes: epoch18 validation loss after update 79 is primary; validation accuracy is secondary. Online training loss is excluded from confirmatory cross-regime inference.
- Accounting: 72 new m=4 branches; approximately 288 ordinary dataset-pass equivalents of new training exposure.

## Frozen seeds and dataset mapping

- Stream master seed (32 raw bytes encoded in hexadecimal): `6a87c34e90f1b52d46e0a9c78d35f12b8e441096c2d7a5ef39b0684d21c59f73`.
- Bootstrap seed: `202609270418`; state-level `Generator(PCG64)` percentile bootstrap with `B=10,000` and common resampling indices.
- Dataset identity path: `internal-artifacts/dataset_identity.json`.
- Dataset identity file SHA-256: `18fcff67679cf832d9398aa8251826b57470d5554de9180ace1fda733a923898`.
- Registered `train_indices_sha256`: `de2fe7734daf05c6820f4906ad4d561661bb952a99549bfcf6c4a61d3bdd113d`.
- Ordered mapping: `local position i -> original unsorted T[i] -> global training example`. The single ordered `T` is common to all 36 states; actual loader position→global-index mapping must match it element by element. Set equality alone is insufficient. Stream 1 is the exact historical Reset A1 order; streams 2–4 use the protocol's frozen HMAC-SHA256 and digest-sort rule over positions of `T`.

## Frozen primary estimands

- `Delta_1`: mean A1-only m=1 LR `0.10 − 0.05` validation-loss contrast across states.
- `Delta_4`: corresponding mean m=4 LR contrast.
- `Psi_4 = Delta_1 − Delta_4`: primary effect-modification estimand.
- `R_.10`: mean m=1 minus m=4 validation loss at LR `0.10`.
- `R_.05`: mean m=1 minus m=4 validation loss at LR `0.05`.
- Required identity: `Psi_4 = R_.10 − R_.05` up to declared floating-point precision.

## Interpretation and stopping boundary

No equivalence margin is defined. A zero-spanning `CI(Psi_4)` means modification is not established at achieved precision. The m=4 intervention changes minibatch-gradient variability, effective gradient-batch construction, and training-example exposure; it is not pure stochasticity removal. Neither deterministic finite-step dynamics `D` nor a unique stochastic mechanism `N` is established by this design. There is no outcome-driven expansion to m=2, m=8, exact-gradient execution, extra replicas, or alternate estimands.

## Execution boundary

**Protocol freeze does not authorize implementation production or scientific execution.** Implementation, its independent numerical-compatibility audit, historical m=1 admission, and separate explicit production execution authorization are later gates. No training or implementation was performed by this freeze administration.

`execution_authorized = false`
