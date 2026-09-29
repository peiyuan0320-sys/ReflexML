<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# ReflexML Phase 5 Preregistered Design Protocol

Protocol version: `Phase5-v1`  
Repository baseline: `0c6d4d1c25286a3fcdaac15ce5c46494367a3f13` (`chore: checkpoint ReflexML through Phase 4A`)  
Primary checkpoint: `t* = 14`  
Protocol status: frozen scientific design, not an execution record

## EXECUTION STATUS

**NOT APPROVED FOR TRAINING**

Reason: the independent raw-artifact backup destination has not yet been specified and verified.

```text
BACKUP_DESTINATION = TBD / EXECUTION BLOCKED
```

No Phase 5 training, checkpoint generation, branching, or outcome inspection may begin until every hard gate in this protocol has passed. In particular, a physically or logically independent backup destination must be named, shown writable, tested with a copied artifact, and verified by SHA-256 equality against the working copy.

## 1. Scope and scientific questions

Phase 5 is a finite-resource signal-validation study of the historical ReflexML learning-rate timing intervention. It is not a controller experiment, deployment study, or mechanism-identification study.

Phase 5 asks two distinct prespecified questions at a single fixed checkpoint epoch:

1. **Phase 5A:** Does the Phase-4A-derived short-horizon response signature replicate in outcome-independent states drawn from the epoch-14 state population?
2. **Phase 5B:** Across those states, is expected short intervention response linearly associated with expected historical final intervention effect?

The corresponding research hypotheses and falsification rules are:

- Phase 5A is supported only through the five-part conjunction defined in Section 4. Any failed component weakens the short-horizon signature and must be reported.
- Phase 5B is evaluated through a two-sided interval for the state-level covariance. An interval spanning zero means that the achieved design did not provide sufficiently precise evidence of the preregistered linear association; it does not prove that every weak association is absent.
- Phase 5A and Phase 5B have no analysis gate. Both are performed and reported whenever evaluable as specified.
- A negative, ambiguous, or non-evaluable result is retained. It does not trigger seed replacement, state reselection, threshold adjustment, a stronger predictor, controller implementation, or Jev integration.

Controlled variables are the historical Fashion-MNIST data and split, model architecture, optimizer and momentum, training protocol, 30-epoch budget, Now/Wait3 intervention, and validation-loss definitions. Only outcome-independent base-training randomness and the preregistered future-shuffle streams vary.

## 2. Frozen experimental setup

The historical reference setup remains frozen:

- dataset: Fashion-MNIST;
- training population: the fixed 10,000-example training subset;
- validation population: the fixed 5,000-example validation subset;
- split seed and exact split fingerprint: the existing project definition;
- model: `Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 10)`;
- loss: cross entropy, recorded as an example-weighted mean;
- optimizer: SGD;
- initial learning rate: `0.1`;
- momentum: `0.9`;
- batch size: `128`;
- training duration: 30 epochs;
- execution reference: CPU with `num_workers=0`, unless an approved new experiment version explicitly changes this before outcomes exist;
- no stochastic augmentation and no dropout;
- official Fashion-MNIST test data are not used for Phase 5 state selection, estimation, or tuning.

The primary checkpoint epoch is frozen at:

```text
t* = 14
```

At this checkpoint, 14 base-training epochs have been completed and 16 epochs remain through epoch 30. Relative horizons `h=1..4` correspond to epochs 15..18. The historical final window, epochs 28..30, corresponds to relative horizons `h=14..16`.

The intervention remains exactly:

- **Now:** use learning rate `0.05` beginning at epoch `t*+1`;
- **Wait3:** use learning rate `0.1` for epochs `t*+1..t*+3`, then `0.05` beginning at epoch `t*+4`;
- both branches remain within the historical epoch-30 budget.

The design does not change the intervention or outcome merely to simplify implementation.

## 3. Target population and independent scientific unit

The primary state population is:

```text
S ~ P_14
```

`P_14` is the epoch-14 checkpoint-state distribution induced by the frozen dataset and split, model, optimizer, training protocol, and the outcome-independent base-training randomness mechanism defined in this protocol. This population is conditional on the project's fixed dataset and split.

The primary independent scientific unit is:

```text
one independent base training run -> one primary epoch-14 state
```

Frozen primary sample size:

```text
N = 36 independent base runs/states
```

Each base run contributes exactly one primary state. Multiple future replicas from one state improve estimation of that state's response but do not create additional independent states.

Phase 3, Phase 3.5, Phase 4, and Phase 4A outcome-selected states are excluded from Phase 5 confirmatory estimation. No Phase 5 state may be selected, ranked, removed, or replaced using future Now/Wait outcomes, `rho`, `tau`, final gain, win rate, future variance, or observed effect direction.

The 36 base seeds are generated by the deterministic outcome-blind rule in Section 9. They are not hand-selected.

They must also be historically non-overlapping. Let `B_P5` be the set of 36 Phase 5 primary base-training seeds and `B_hist` the outcome-blind registry of base-training seeds actually used in completed historical ReflexML phases. The preregistered identity rule is:

```text
B_P5 ∩ B_hist = ∅
```

`B_hist` is an identity/provenance registry, not an outcome-derived selection device. Its construction, reliability requirement, and collision handling are specified in Sections 8 and 9.

## 4. Phase 5A estimands and claim rules

For state `S` and future trajectory `omega`, define the paired validation-loss difference:

```text
Delta L_h(S, omega) = L_W(t*+h) - L_N(t*+h)
```

Positive values favor Now at that horizon.

Define the per-replica short response:

```text
d(S, omega) = (Delta L_1 + Delta L_2 + Delta L_3) / 3
```

The frozen state-level short-response estimand is:

```text
rho(S) = E_omega[d(S, omega) | S]
```

The Phase 5A population primary estimand is:

```text
mu_rho = E_{S~P_14}[rho(S)]
```

The expected historical direction is `mu_rho > 0`.

The key confirmatory secondary estimand is:

```text
A_34 = E_{S,omega}[Delta L_4 - Delta L_3]
```

Its expected historical direction is `A_34 < 0`. It must be called the **preregistered h=3->4 additive boundary contrast**. A negative estimate is described as attenuation only when the observed h=3/h=4 pattern actually moves toward zero. The historical Phase 4A attenuation ratio is not a Phase 5 confirmatory estimand.

The preregistered horizon-level directional diagnostics are:

```text
mu_h = E_{S,omega}[Delta L_h],  h = 1,2,3
```

They test directional concordance with the short-horizon signature; they are not three additional co-primary CI tests.

### 5A claim rule

The exact phrase **5A signature supported** may be used if and only if all five conditions hold:

1. the two-sided 95% CI for `mu_rho` lies entirely above zero;
2. the two-sided 95% CI for `A_34` lies entirely below zero;
3. the point estimate of `mu_1` is positive;
4. the point estimate of `mu_2` is positive;
5. the point estimate of `mu_3` is positive.

Every other evaluable outcome is classified as **5A signature not fully supported**, with every failed component stated explicitly.

If the two summary-estimand intervals meet their criteria but any individual `mu_h` point estimate is non-positive, the required wording is:

> summary estimands supported, but horizon-level directional concordance was not fully reproduced

Preferred description of the target is **replication of the preregistered Phase-4A-derived short-horizon signature**. The broader phrase “full Phase 4A replication” is prohibited.

Phase 5A confirmatory analysis uses A-block replicas only. B-block h=1..4 records may not be pooled into any 5A confirmatory estimator.

## 5. Phase 5B estimands and claim rules

For B-block replica `r`, let `L_N,r` and `L_W,r` be the Now and Wait mean validation losses across epochs 28..30. The historical per-replica gain is:

```text
G_r = (L_W,r - L_N,r) / max(L_W,r, 1e-12)
```

The frozen state-level final-effect estimand is:

```text
tau(S) = E_omega[G(S, omega) | S]
```

`tau` is the **expectation of the per-replica ratio**. It is not:

```text
(E[L_W] - E[L_N]) / E[L_W]
```

This distinction must be retained in the schema, estimator, tests, and report.

The Phase 5B primary association estimand is:

```text
psi = Cov_{S~P_14}(rho(S), tau(S))
```

For the primary estimator:

- `rho_hat_i` is estimated only from state `i`'s A block;
- `tau_hat_i` is estimated only from state `i`'s B block;
- A and B future replicas are disjoint;
- B-block short-horizon observations never enter the primary `rho_hat_i`.

The scientific description is **fixed-epoch state-level response-effect association**.

### 5B claim rule

- If the two-sided 95% CI for `psi` lies entirely above zero, report a **positive state-level response-effect association**.
- If it lies entirely below zero, report a **negative state-level association: stronger positive short response corresponds to weaker or more negative historical final intervention effect**.
- If it spans zero, report **no sufficiently precise evidence of the preregistered linear state-level association at the achieved Phase 5 precision**.

An interval spanning zero must not be summarized as “there is no association.” The design was not chosen to exclude arbitrarily weak associations.

`psi` is not causal surrogacy, deployment-time predictability, incremental information beyond arbitrary pre-intervention `X_t`, a controller-ready feature, or evidence that `rho` is observable before intervention.

## 6. Heterogeneity estimand and estimator

The frozen final-effect heterogeneity estimand is:

```text
sigma_tau^2 = Var_{S~P_14}[tau(S)]
```

The practical reference is:

```text
sigma_tau,practical   = 0.005
sigma_tau,practical^2 = 2.5e-5
```

This is a practical reference line, not a mechanical success, failure, continuation, or stopping gate.

For state `i`, using its `K_B=10` B-block replicas:

```text
tau_hat_i = (1/K_B) * sum_r G_ir

s_tau,w,i^2 = (1/(K_B-1)) * sum_r (G_ir - tau_hat_i)^2

Var_hat(tau_hat_i | S_i) = s_tau,w,i^2 / K_B
```

Let `S^2(tau_hat_i)` denote the sample variance across the 36 state-level estimates, using denominator `N-1`. The frozen deconvolved estimator is:

```text
sigma_tau^2_hat
  = S^2(tau_hat_i)
    - (1/N) * sum_i [s_tau,w,i^2 / K_B]
```

The naive `S^2(tau_hat_i)` is not the Phase 5 heterogeneity estimator.

If the raw deconvolved estimate is negative because of finite-sample noise:

- retain the negative value for inferential reporting and bootstrap construction;
- interpret it as failure to resolve positive between-state heterogeneity at the achieved precision;
- do not truncate before inference.

For descriptive display only, the report may include:

```text
sigma_tau,display = sqrt(max(0, sigma_tau^2_hat))
```

The report must show the raw deconvolved variance estimate, its uncertainty interval, an optional descriptive SD, and its relationship to the `0.005` reference without converting that reference into a post hoc hard gate.

## 7. A/B replica architecture

Frozen replica depths per state are:

```text
K_A = 15 future replica pairs
K_B = 10 future replica pairs
```

### A block

- Runs only relative horizons `h=1..4`, corresponding to epochs 15..18.
- Provides all Phase 5A confirmatory quantities.
- Provides the primary `rho_hat_i` used in `psi`.
- Does not provide primary `tau_hat_i`.

### B block

- Runs from the restored epoch-14 checkpoint through epoch 30.
- Provides the epoch 28..30 losses, per-replica `G`, `tau_hat_i`, and `sigma_tau^2` inputs.
- Its h=1..4 records are retained but are excluded from Phase 5A confirmatory estimators and from the primary `rho_hat_i`.
- B-block h=1..4 may be used only for the prespecified secondary internal-replication sensitivity.

A and B replica identities and future streams must be disjoint. A/B labels refer to future-replica blocks and must not be confused with the Now and Wait action branches inside every replica pair.

Across 36 states, the primary design contains:

- 540 A-block replica pairs;
- 360 B-block replica pairs;
- 900 total paired future trajectories;
- 900 Now branches and 900 Wait branches.

## 8. Randomness, CRN, and future-stream independence

Within every Now/Wait pair, common random numbers are mandatory. Before intervention, the two branches must have identical:

- checkpoint identity and hash;
- model state;
- optimizer state and momentum;
- data split and model architecture;
- relevant global RNG states;
- DataLoader generator state and intended future stream;
- configuration other than the predefined LR timing intervention.

During execution, the actual minibatch indices and batch boundaries must match between Now and Wait at every paired epoch. Different LR values may subsequently produce different parameters, gradients, momentum values, losses, and accuracies; those differences are part of the intervention effect.

Across A and B, future replicas are disjoint. Across primary states, the same future random stream may not be reused.

The required conceptual mapping is injective:

```text
(phase_version, base_run_id, block, replica_id)
    -> future seed / future stream
```

All 936 Phase-5-generated seed integers—36 base seeds plus 900 future seeds—must be globally unique within Phase 5.

Integer inequality alone is insufficient. The execution audit must also compare actual future-order records or cryptographic hashes across state/block/replica assignments so accidental stream or order reuse cannot pass silently.

No pre-existing general-purpose protected/excluded seed registry existed at protocol-authoring time. Phase 5 nevertheless requires a narrowly defined historical **base-training seed registry** for the non-overlap rule in Section 3. That registry must be derived solely from completed repository artifacts, manifests, and provenance fields that identify base training runs. Historical outcome magnitude, effect direction, gain, win rate, future variance, and all other outcome-dependent fields are prohibited inputs to registry construction.

Repository evidence at protocol-hardening time is sufficient to support a reliable registry: the completed Phase 3 design, 120-row dataset, and audits identify base-training seeds 42..61, which contain the base seeds used by earlier completed phases. The formal registry must still be materialized before the Phase 5 base-seed manifest, with its exact source paths, source hashes, derivation procedure, entries, and registry SHA-256 recorded. Missing historical seeds may not be inferred or invented. If the repository no longer supports a reliable registry when execution is prepared, that is an unresolved execution hard gate.

Historical future-replica seeds are not part of `B_hist` and do not require exclusion unless a separately approved repository policy explicitly requires it.

As a second, outcome-blind identity safeguard, let `H_P5` be the set of SHA-256 content hashes of the 36 Phase 5 primary epoch-14 checkpoints and `H_hist` the reliably identified content hashes of checkpoints from completed historical artifacts. Before any Phase 5 branching outcome exists, the audit must verify:

```text
H_P5 ∩ H_hist = ∅
```

At protocol-hardening time, the Phase 4A protected input manifest records 180 historical `.pt` paths and 160 unique SHA-256 checkpoint hashes, providing a reliable basis for `H_hist` when restricted to completed artifacts. The formal registry must record its scope and provenance. The checkpoint-hash audit may not use historical outcome values.

## 9. Base and future seed-generation rule

Seed generation is deterministic and outcome-blind. Namespaces are encoded as UTF-8 exactly as shown, with no surrounding whitespace or newline.

For any namespace string:

1. compute `SHA256(namespace.encode("UTF-8"))`;
2. interpret the first 8 digest bytes as an unsigned big-endian integer `x`;
3. compute `seed = 1 + (x mod (2^31 - 2))`;
4. begin with `counter=0`;
5. if the candidate violates a required collision rule, increment the counter by one and recompute.

This yields a positive integer in `1..2^31-2`, compatible with the current project's integer seed path. No incompatible seed constraint was found in the repository at protocol-authoring time.

### Base seeds

For `index=1..36`, use:

```text
ReflexML|Phase5|v1|base|<index>|<counter>
```

`<index>` and `<counter>` use their ordinary base-10 representations without zero padding. The stable `base_run_id` used by the future namespace is the same base-10 index `1..36`.

Before deriving these 36 seeds, construct and freeze `B_hist` as specified in Section 8. For a base-seed candidate, reject and deterministically increment `<counter>` if the candidate:

- duplicates any earlier Phase 5 generated seed;
- belongs to `B_hist`;
- belongs to any separately approved explicit repository-defined exclusion registry that actually exists.

Thus Phase 5 base seeds are both globally unique within Phase 5 and disjoint from historical base-training seeds.

### Future seeds

For each state, block `A` or `B`, and one-based replica ID, use:

```text
ReflexML|Phase5|v1|future|<base_run_id>|<block>|<replica_id>|<counter>
```

`<block>` is exactly the uppercase ASCII character `A` or `B`. IDs and counters use ordinary base-10 representations without zero padding.

For a future-seed candidate, collision rejection applies to every previously generated Phase 5 base or future seed and to a separately approved explicit repository policy only if that policy expressly covers future seeds. `B_hist` is a base-seed exclusion set and is not silently extended to historical future-replica seeds.

Generation order is fixed so collision resolution is reproducible: generate base seeds in ascending `index=1..36`; then generate future seeds in ascending `base_run_id`, block order `A` then `B`, and ascending `replica_id` within each block. Every collision check uses the set accumulated earlier in this order.

Before any Phase 5 base training begins, `B_hist` and its provenance/hash must be frozen, the complete base-seed manifest must be generated, and `B_P5 ∩ B_hist = ∅` must be verified. Before any Phase 5 branch outcome exists, the complete future mapping must also be generated, checked for Phase-5-global uniqueness and any applicable explicit exclusions, hashed, reviewed, and committed with the design version and Git baseline. Once execution begins, the mapping is immutable.

After the 36 authorized base runs produce their epoch-14 checkpoints—but before any A- or B-block branch is run—construct the reliably scoped `H_hist` registry and verify `H_P5 ∩ H_hist = ∅`. A collision is an identity/integrity failure requiring investigation; it is not resolved by consulting outcomes or silently replacing a completed state.

## 10. Divergence, technical failure, and retry policy

Technical failure and substantive algorithmic divergence are different events and must be recorded separately.

### Technical failure

Examples include:

- machine or process interruption;
- corrupted or incomplete write;
- checkpoint, source, or manifest hash mismatch;
- RNG restoration or audit failure;
- schedule mismatch;
- paired minibatch-order mismatch;
- missing or duplicate required epoch;
- output-path collision or mixed experiment version.

A technical failure may be retried only with exactly the same base run ID, checkpoint, block, replica ID, and future seed/stream mapping. The retry ledger is append-only. A failed attempt may never be replaced by a newly generated seed.

Systemic integrity failures stop Phase 5 execution until investigated. If a scientifically meaningful implementation or protocol change is needed, it creates a new experiment version rather than silently modifying `phase5_v1`.

### Substantive divergence

Substantive divergence is a non-finite model/training state or required outcome produced by an otherwise audit-valid execution. A finite but extreme loss or `G` is a valid observation and is not divergence.

Substantive divergence must not be deleted, replaced, assigned an arbitrary penalty, or clipped into the historical continuous estimand. Now and Wait divergence are recorded separately, including state, block, replica, epoch, action, and audit status.

If any required A-block outcome through h=4 is non-finite:

- Phase 5A continuous confirmatory analysis is **NON-EVALUABLE AS SPECIFIED**;
- primary `psi` is also **NON-EVALUABLE AS SPECIFIED**, because its `rho` is A-derived.

If A is evaluable but any required B-block final `G` is non-finite:

- continuous `tau`, `sigma_tau^2`, and primary `psi` are **NON-EVALUABLE AS SPECIFIED**.

All reports must include action-specific divergence counts and rates plus state and replica locations.

A finite-pair quantity such as

```text
tau_finite = E[G | D_N=0, D_W=0, S]
```

may be reported only as a conditional sensitivity analysis. It does not replace historical `tau` and cannot support a 5B primary claim.

If an apparent substantive divergence does not reproduce under an identical deterministic rerun, the discrepancy is an experiment-integrity failure requiring investigation, not permission to remove an outlier.

## 11. Statistical estimators

For state `i` and A-block replica `r`, retain the complete vector:

```text
(Delta L_i,r,1, Delta L_i,r,2, Delta L_i,r,3, Delta L_i,r,4)
```

Define:

```text
d_i,r = (Delta L_i,r,1 + Delta L_i,r,2 + Delta L_i,r,3) / 3

rho_hat_i = (1/K_A) * sum_r d_i,r

mu_rho_hat = (1/N) * sum_i rho_hat_i

A_34_hat
  = (1/N) * sum_i [(1/K_A) * sum_r
      (Delta L_i,r,4 - Delta L_i,r,3)]

mu_h_hat
  = (1/N) * sum_i [(1/K_A) * sum_r Delta L_i,r,h]
```

For B-block replicas:

```text
tau_hat_i = (1/K_B) * sum_r G_i,r
```

Let `rho_bar` and `tau_bar` be the means of the 36 state-level estimates. The primary sample-covariance estimator is:

```text
psi_hat
  = (1/(N-1)) * sum_i
      [(rho_hat_i - rho_bar) * (tau_hat_i - tau_bar)]
```

A/B disjointness prevents shared future-replica noise from mechanically creating the primary state-level covariance. The primary analysis does not standardize `psi` into a correlation and does not adjust it for arbitrary pre-intervention covariates.

The heterogeneity estimator is exactly the deconvolved estimator in Section 6. Replicas are never counted as independent states. Missing, non-finite, or divergent primary quantities follow Section 10 rather than an available-case substitution invented after outcomes.

## 12. Bootstrap and uncertainty procedure

The primary uncertainty framework is a nested state-level bootstrap with:

```text
B_boot = 20,000 bootstrap replicates
```

The primary bootstrap RNG is frozen as:

```text
RNG implementation: NumPy Generator(PCG64)
seed namespace:     ReflexML|Phase5|v1|bootstrap|primary
```

Encode the namespace exactly as UTF-8 with no surrounding whitespace or newline, compute SHA-256, interpret the first 8 digest bytes as one unsigned big-endian integer, and use that integer directly as the primary bootstrap seed. For audit convenience, the frozen derivation is:

```text
SHA-256:    94778b440d5865af8a3d6d482f184d7eb1c40cde8aaa65a3f3134ad2c289f770
first 8:    94778b440d5865af
integer:    10698172564239836591
```

The analysis manifest must record the derived integer seed, NumPy version, bit-generator name `PCG64`, and protocol version. The implementation must reproduce this derivation exactly; the RNG algorithm and seed are not TBD.

### Outer bootstrap

For each bootstrap replicate, sample 36 primary states with replacement.

If an original state appears multiple times, each appearance is a separate bootstrap occurrence. Every occurrence receives its own independent inner A resample and independent inner B resample. Inner resamples may not be reused merely because two outer occurrences point to the same original state.

### Inner A bootstrap

For every outer-state occurrence, sample `K_A=15` A-block replica pairs with replacement. The entire four-horizon trajectory remains one resampled unit. Horizons may not be resampled independently.

### Inner B bootstrap

Independently sample `K_B=10` B-block replica pairs with replacement for each outer-state occurrence. The complete B record—including final `G`, final branch metrics, and associated paired provenance—remains one unit.

A and B inner resampling are independent conditional on each outer-state occurrence.

Every bootstrap replicate recomputes at least:

- `mu_rho_hat`;
- `A_34_hat`;
- `mu_1_hat`, `mu_2_hat`, and `mu_3_hat`;
- `psi_hat`;
- raw deconvolved `sigma_tau^2_hat`.

Negative variance-component estimates remain negative inside the bootstrap distribution. They are not truncated before CI construction.

Primary intervals are two-sided 95% percentile bootstrap intervals using the empirical 2.5% and 97.5% quantiles of the 20,000 replicate estimates.

## 13. Multiplicity and interpretation rules

No post hoc Holm family is constructed across all Phase 5 quantities.

Phase 5A and Phase 5B are distinct prespecified scientific questions:

- The Phase 5A claim is the five-component conjunction in Section 4. The three horizon-level diagnostics use point-estimate direction only and are not promoted into three additional CI-based confirmatory hypotheses.
- Phase 5B separately reports `psi` and its two-sided 95% CI.
- `sigma_tau^2` is reported with uncertainty and a practical reference, not as a third binary gate.

There is no 5A-to-5B analysis gate. There is no cross-question omnibus claim and no phrase such as “Phase 5 overall significant.”

### Interpretation matrix

| 5A result | 5B result | Required interpretation |
|---|---|---|
| Signature supported | Positive association supported | The preregistered short-horizon signature replicates on independent states and a positive fixed-epoch state-level short/final association is observed. This may justify a later pre-intervention predictability study. |
| Signature supported | Negative association supported | The short-horizon signature replicates, but the association is opposite to a straightforward positive-surrogate-like interpretation. |
| Signature supported | CI spans zero | Short-horizon dynamics generalize, but there is insufficient evidence that the chosen short response tracks historical final effect at the achieved precision. |
| Signature not fully supported | Positive or negative association supported | A short/final association may exist, but it is not confirmation of the Phase-4A-derived short-horizon signature. |
| Signature not fully supported | CI spans zero | The current short-response-to-final-effect route lacks support at the achieved precision. |

If a relevant continuous primary analysis is **NON-EVALUABLE AS SPECIFIED**, it is not forced into this matrix.

## 14. Secondary and sensitivity analyses

The following are explicitly non-primary:

- B-block h=1..4 internal replication sensitivity;
- other checkpoint epochs and epoch-stratified robustness;
- fixed-relative-horizon final outcomes;
- rank association;
- alternative historical final metrics;
- latent or errors-in-variables models;
- standardized latent correlation;
- hierarchical repeated-measures models;
- optional A/B swap or cross-fit-inspired sensitivity;
- finite-pair `tau_finite` under substantive divergence.

Raw pooling of multiple checkpoint epochs as independent states is prohibited. Secondary checkpoints from one base run remain linked repeated measures.

No secondary or latent model may rescue, replace, or relabel a failed, ambiguous, or non-evaluable primary result. Exact secondary specifications that are not frozen before Phase 5 outcomes are inspected are labeled **exploratory**.

At protocol version 1, the optional A/B swap, secondary checkpoint epochs, fixed-relative-horizon sensitivity, and latent/EIV model remain TBD and are not approved for execution.

## 15. Data schema and provenance requirements

Phase 5 must persist sufficient raw and derived records to reconstruct every estimator without relying on mutable conversation history.

### Design and environment

- protocol version and protocol SHA-256;
- Git commit and clean/dirty status at execution authorization;
- code and test hashes;
- Python, PyTorch, torchvision, NumPy, platform, device, thread, and DataLoader-worker details;
- full frozen training/intervention configuration;
- data source, split seed, exact train/validation indices fingerprint, and dataset-file integrity metadata;
- historical base-training seed registry entries, source/provenance paths and hashes, derivation method, and registry SHA-256;
- historical checkpoint-hash registry scope, source/provenance, entries, and registry SHA-256;
- seed-generation specification and manifest hashes;
- primary bootstrap derived integer seed, NumPy version, bit-generator name `PCG64`, and protocol version.

### Base runs and primary states

- `base_run_id`, index, base seed, derivation namespace/counter, status, and attempt lineage;
- full epoch 1..14 base metrics;
- epoch-14 checkpoint path, content hash, epoch, config, split fingerprint, model/optimizer/RNG/loader-state presence checks;
- explicit statement that the state entered the manifest before future outcomes existed;
- one-primary-state-per-base-run audit;
- base-seed non-overlap audit against `B_hist`;
- primary-checkpoint SHA-256 non-overlap audit against `H_hist`.

### Replica registry

- phase version, base run ID, checkpoint hash, block, replica ID, future seed, namespace/counter, and mapping-manifest hash;
- globally unique mapping key and seed;
- technical attempt and retry identifiers;
- intended and actual epoch/horizon coverage.

### Branch and paired metrics

- action branch (`Now` or `Wait`), epoch, relative horizon, learning rate, train loss, validation loss, and validation accuracy;
- actual minibatch indices including batch boundaries, or lossless protected records plus per-epoch hashes;
- initial-state equality checks, intervention-only schedule difference, paired RNG/order checks, and source-checkpoint immutability checks;
- A records for `Delta L_1..Delta L_4`, per-replica `d`, and `Delta L_4-Delta L_3`;
- B records for epoch 28, 29, and 30 branch losses, each branch's three-epoch mean, and per-replica `G`;
- explicit formula/version fields that distinguish expectation of per-replica ratios from ratio of expected losses.

### Failure, divergence, and integrity records

- append-only technical failure/retry ledger;
- action-specific substantive-divergence records;
- exclusions, if any, with predeclared reason codes—never deletion;
- input, working-output, and backup checksum manifests;
- final row counts, uniqueness checks, missingness checks, schedule checks, and no-overwrite audit.

Small manifests, mappings, schema definitions, integrity metadata, code, tests, and reports belong in Git. Large raw experimental artifacts remain outside normal Git tracking under the approved working and independent backup locations.

No API key, `.env` secret, relay credential, GitHub token, authentication header, or private key may appear in Git, seed manifests, checksum manifests, logs, or backup bundles.

## 16. Integrity tests required before execution

Implementation-specific tests must exist and pass before training authorization. At minimum they must verify:

1. outcome-blind reconstruction of the historical base-training seed registry from completed artifacts and exact reproduction of its SHA-256;
2. refusal to guess if historical base-seed provenance is incomplete or unreliable;
3. deterministic reproduction of all base seeds from the frozen namespaces and counters;
4. `B_P5 ∩ B_hist = ∅`;
5. deterministic reproduction of all future seeds and mapping rows;
6. Phase-5-global uniqueness of all 936 generated seed integers;
7. compliance with any separately approved explicit seed-exclusion registry, if one exists and applies;
8. A/B replica-ID and seed disjointness;
9. one primary epoch-14 state per base run and no outcome-dependent selection path;
10. outcome-blind reconstruction of the completed historical checkpoint-hash registry and `H_P5 ∩ H_hist = ∅` before branching;
11. exact checkpoint restoration of model, optimizer/momentum, global RNG, loader generator, history, config, and split fingerprint;
12. source checkpoint immutability during branching;
13. Now/Wait initial-state equality and LR-only intervention difference;
14. exact within-pair minibatch-order equality at every epoch;
15. different future streams across replicas and no accidental cross-state/block order reuse;
16. exact repeated execution for the same frozen mapping;
17. A-block schedule and exact h=1..4 stopping boundary;
18. B-block schedule and complete epoch 15..30 coverage;
19. correct h-to-absolute-epoch indexing and epoch 28..30 reconstruction;
20. expectation-of-per-replica-ratio semantics, including a counterexample to ratio of means;
21. A-only construction of 5A estimators and primary `rho_hat_i`;
22. B-only construction of primary `tau_hat_i`;
23. correct sample covariance across states without treating replicas as states;
24. correct `sigma_tau^2` deconvolution and retention of negative raw estimates;
25. exact bootstrap namespace derivation and use of NumPy `Generator(PCG64)` with seed `10698172564239836591`;
26. nested bootstrap treatment of repeated outer-state occurrences with independent inner A and B resamples;
27. whole A trajectories and whole B records remaining intact under resampling;
28. negative bootstrap variance components remaining untruncated;
29. technical failure retry with identical IDs and seed mapping;
30. substantive-divergence paths producing the required non-evaluable status;
31. finite extremes remaining valid observations;
32. append-only failure/retry behavior and output-directory no-overwrite protection;
33. working-copy and backup checksum-manifest construction and verification.

Existing historical tests are evidence for the underlying checkpoint and pairing mechanisms, but they do not replace these Phase 5-specific tests.

## 17. Artifact backup and immutability policy

Git stores the frozen protocol, implementation and analysis code, tests, small manifests, seed mappings, schema/version definitions, checksum metadata, and final reports.

Large raw experimental artifacts use:

```text
working copy
+ independent backup copy
+ cryptographic checksum manifest
```

The intended working experiment namespace is versioned, for example:

```text
runs/phase5_v1/
```

Existing experiments and completed attempts are append/version only and are never silently overwritten. A scientifically meaningful protocol or implementation change creates a new experiment version. Failure and retry records are append-only.

The exact backup destination is intentionally unresolved:

```text
BACKUP_DESTINATION = TBD / EXECUTION BLOCKED
```

The destination must be physically or logically independent of the working artifact location. Naming a second path on the same failure domain is not automatically sufficient; its independence must be documented during approval.

## 18. Execution hard gates

Phase 5 remains blocked until all of the following are true:

- this protocol has been reviewed and approved;
- the frozen protocol is committed before Phase 5 outcomes exist;
- the repository commit and design version are recorded;
- the historical base-training seed registry is constructed outcome-blindly from reliable completed-artifact provenance, recorded, and hashed;
- the base-seed manifest is generated deterministically;
- Phase 5 primary base-seed non-overlap with the historical registry is verified;
- the future-seed mapping is generated deterministically;
- Phase-5-global seed uniqueness is verified;
- any separately approved explicit repository-defined seed exclusions are checked if they exist and apply;
- A/B block disjointness is verified;
- the seed/mapping manifests are hashed, reviewed, and committed;
- the implementation is verified to use NumPy `Generator(PCG64)` and the frozen primary bootstrap seed derivation;
- implementation-specific integrity tests are written and pass;
- expectation-of-per-replica-ratio semantics are tested;
- `sigma_tau^2` deconvolution is tested;
- divergence and non-evaluable paths are tested;
- append-only/no-overwrite behavior is tested;
- an independent backup destination is named and documented;
- the destination is writable;
- a test artifact is copied to it;
- SHA-256 equality between working and backup copies is verified;
- no unresolved hard gate remains.

Only after these pre-base-training gates pass may a separate authorization approve the 36 frozen base runs. Approval of this document alone does not execute or start Phase 5.

After authorized base training produces the 36 epoch-14 primary checkpoints, but before any A- or B-block branch is run, a second execution gate requires:

- reliable outcome-blind construction and hashing of the completed historical checkpoint-hash registry;
- verification that Phase 5 primary checkpoint hashes do not overlap historical checkpoint hashes;
- checkpoint restoration verification;
- Now/Wait CRN order-equality verification on the required integrity path;
- cross-state future-stream reuse checks implemented and ready;
- no unresolved identity or integrity failure.

Only after this second gate passes may a separate authorization approve Phase 5 branching and outcome generation.

## 19. Computational budget

Frozen primary design:

```text
t*  = 14
N   = 36
K_A = 15
K_B = 10
```

Each A pair runs two branches for four epochs:

```text
2 * 4 = 8 branch-epochs
```

Each B pair runs two branches for 16 epochs:

```text
2 * (30 - 14) = 32 branch-epochs
```

Primary branch cost:

```text
36 * (8*15 + 32*10) = 15,840 branch-epochs
```

Base-run training:

```text
36 * 14 = 504 base epochs
```

Approximate primary total:

```text
15,840 + 504 = 16,344 epoch-equivalents
```

This excludes technical retries, tests, optional secondary checkpoints, optional A/B swap, and other sensitivity extensions. Technical retries reproduce the same frozen mapping and do not increase the scientific sample size.

The design is a finite-resource signal-validation design and is not powered to exclude arbitrarily weak latent association or heterogeneity.

## 20. Explicit non-claims and scope limitations

Regardless of outcome, Phase 5 does not by itself establish:

- causal surrogacy of short response for final intervention effect;
- deployment-time predictability;
- controller value or an actionable policy;
- that `rho` is available before taking the intervention;
- incremental predictive information beyond arbitrary pre-intervention state features;
- SGD mixing, stationarity, or ergodicity;
- a universal decay, attenuation, or memory constant;
- generalization beyond the fixed Fashion-MNIST task, model, optimizer, data subset, and split;
- independence with respect to resampled datasets or validation populations;
- absence of a weak association when the Phase 5B interval spans zero.

All primary states share the same fixed dataset and split. The design has limited sensitivity to weak state-level association and weak between-state heterogeneity.

A positive `psi` would justify considering a later, separately preregistered study of pre-intervention predictors such as `X_t -> rho` or `X_t -> tau`. It would not justify immediate deployment or direct optimizer control.

## 21. Frozen versus still-TBD decisions

### Frozen

- `t*=14`, `N=36`, `K_A=15`, and `K_B=10`;
- `P_14` and one-base-run-to-one-primary-state rule;
- outcome-independent deterministic base-state sampling;
- historical base-state non-overlap through an outcome-blind base-seed registry and primary-checkpoint hash audit;
- historical Now/Wait3 intervention and within-pair CRN;
- A-only Phase 5A and A-only primary `rho_hat_i`;
- B-only primary `tau_hat_i`;
- disjoint A/B future replicas;
- `rho`, `mu_rho`, `A_34`, and `mu_1..mu_3` definitions;
- historical per-replica-ratio `tau`;
- covariance `psi` primary association;
- `sigma_tau^2`, its deconvolved estimator, and the `0.005` practical reference;
- Phase-5-global seed uniqueness and SHA-256 seed-derivation rule;
- divergence, technical-failure, retry, and non-evaluable policies;
- nested state-level bootstrap with 20,000 replicates using NumPy `Generator(PCG64)` and the frozen SHA-256-derived primary seed;
- independent inner resampling for every outer-state occurrence;
- whole-trajectory/whole-record resampling units;
- percentile two-sided 95% intervals;
- 5A conjunction, separate 5B claim, no analysis gate, and no omnibus Phase 5 claim;
- append/version artifact policy and verified independent backup as an execution hard gate.

### Still TBD or not approved for execution

- exact physical/logical independent backup destination;
- whether optional A/B swap is executed;
- exact secondary checkpoint epochs;
- exact fixed-relative-horizon sensitivity definition;
- exact latent/EIV sensitivity model;
- any secondary analysis not frozen before outcomes are inspected.

Unfrozen secondary work is exploratory. No TBD item may be resolved after inspecting relevant Phase 5 outcomes and then presented as preregistered.

## 22. Final execution checklist

This is a staged execution checklist. Items through explicit base-run authorization must pass before base training; the remaining identity and branching items must pass before any A/B outcome generation. At protocol-hardening time the missing backup destination blocks even the first stage.

- [ ] `PHASE5_DESIGN.md` reviewed and approved.
- [ ] Frozen design committed before Phase 5 outcomes exist.
- [ ] Exact execution commit and design version recorded.
- [ ] Historical base-training seed registry constructed outcome-blindly from completed artifacts, with sources, provenance, entries, and SHA-256 recorded.
- [ ] Historical base-seed registry confirmed reliable; otherwise execution remains blocked without guessing.
- [ ] Deterministic 36-row base-seed manifest generated.
- [ ] `Phase5 base seeds ∩ historical base-training seeds = ∅` verified.
- [ ] Deterministic 900-row future-seed mapping generated.
- [ ] All 936 Phase 5 seed integers verified globally unique.
- [ ] Any separately approved explicit repository-defined seed exclusions checked, if such a registry exists and applies.
- [ ] A/B block IDs and streams verified disjoint.
- [ ] Seed and mapping manifests hashed, reviewed, and committed.
- [ ] Primary bootstrap implementation verified as NumPy `Generator(PCG64)` with namespace-derived seed `10698172564239836591`.
- [ ] Phase 5 implementation-specific integrity tests written and passing.
- [ ] Independent backup destination named and independence documented.
- [ ] Backup destination shown writable.
- [ ] Test artifact copied to the backup destination.
- [ ] Working and backup test-artifact SHA-256 values shown equal.
- [ ] Separate explicit authorization for the 36 frozen base runs recorded.
- [ ] After base training, historical checkpoint-hash registry constructed outcome-blindly from completed artifacts, with scope, provenance, entries, and SHA-256 recorded.
- [ ] `Phase5 primary checkpoint hashes ∩ historical checkpoint hashes = ∅` verified before branching.
- [ ] Complete checkpoint restoration verified.
- [ ] Now/Wait CRN and actual minibatch-order equality verified.
- [ ] Cross-state and cross-block future-stream reuse checks verified.
- [ ] Expectation-of-per-replica-ratio semantics tested.
- [ ] `sigma_tau^2` deconvolution and negative-estimate handling tested.
- [ ] Technical failure, identical retry, substantive divergence, and non-evaluable paths tested.
- [ ] Append-only ledger and no-overwrite behavior tested.
- [ ] Working-output checksum-manifest process verified.
- [ ] No unresolved execution hard gate remains.
- [ ] Separate explicit authorization for A/B branching and outcome generation recorded.

Current execution status remains **NOT APPROVED FOR TRAINING**. After the backup and all pre-base-training gates pass, base training still requires separate explicit authorization; completion of base training does not authorize branching until every post-base identity/integrity gate also passes.

## FROZEN DECISIONS

| Domain | Frozen decision |
|---|---|
| Protocol | `Phase5-v1`; preserve historical phases and outputs |
| Primary state population | `S ~ P_14`, conditional on the fixed dataset/split |
| Independent unit | One outcome-independent base run -> one epoch-14 state |
| Scale | `N=36`, `K_A=15`, `K_B=10` |
| Intervention | Historical Now `.05` immediately vs Wait3 `.1` for h=1..3 then `.05` |
| Pairing | CRN and identical actual minibatch ordering within each Now/Wait pair |
| A block | h=1..4 only; sole confirmatory source for 5A and primary `rho_hat_i` |
| B block | h=1..16/epoch30; sole primary source for `tau_hat_i` |
| 5A primary | `mu_rho = E_S[rho(S)]` |
| 5A key secondary | `A_34 = E[Delta L_4-Delta L_3]` |
| 5A signature | Two directional CI criteria plus positive point estimates for `mu_1..mu_3` |
| Final effect | `tau(S)=E_omega[(L_W-L_N)/max(L_W,1e-12) | S]` |
| 5B primary | `psi=Cov_S(rho(S),tau(S))`, A/B-disjoint estimator |
| Heterogeneity | Deconvolved `sigma_tau^2`; raw negative estimates retained |
| Practical reference | `sigma_tau=0.005`, descriptive reference rather than gate |
| Seeds | Deterministic SHA-256 namespaces; all 936 Phase 5 seeds globally unique; 36 base seeds disjoint from outcome-blind historical base-seed registry |
| State identity | Phase 5 primary checkpoint SHA-256 set disjoint from reliably identified historical checkpoint hashes before branching |
| Failure policy | Same-mapping retry for technical failure; substantive divergence retained |
| Non-evaluability | Required non-finite A or B outcomes trigger the specified primary non-evaluable status |
| Uncertainty | Nested state-level bootstrap, 20,000 replicates, NumPy `Generator(PCG64)`, frozen namespace-derived seed, two-sided 95% percentile CI |
| Multiplicity | Separate 5A conjunction and 5B CI; no analysis gate or overall omnibus claim |
| Secondary analyses | Cannot rescue primary; unfrozen specifications are exploratory |
| Artifacts | Version/append, no overwrite; working copy + independent backup + SHA-256 manifest |
| Current execution status | **NOT APPROVED FOR TRAINING**; backup destination remains unverified TBD |
