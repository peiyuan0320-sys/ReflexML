<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# ReflexML Gradient-Noise Attenuation: frozen prospective protocol

**Protocol ID:** `ReflexML-Gradient-Noise-Attenuation-v1`

**Status:** `FROZEN — EXECUTION NOT AUTHORIZED`

**Freeze state:** `FROZEN`

**Execution authorized:** `false`

**Pre-freeze audited candidate SHA-256:** `8d41c5525876e08c5d45b39056dd85f7a5be20440ab5807ff822220091d93cb2`

**Scientific scope:** One-epoch, Reset-S17, controlled four-gradient averaging against the existing ordinary-SGD Reset A1 reference. This frozen protocol requires later independent implementation audit and separate execution authorization. It grants neither.

## 1. Scientific question and prior evidence

**Question:** Does controlled attenuation of minibatch-gradient variability causally modify the immediate epoch18 LR `0.05` validation-loss advantage relative to LR `0.10`?

The closed mechanism-stage synthesis is `MOMENTUM_RESET_MECHANISM_CLOSURE.md`, SHA-256 `495fa3363d1631e5efdfea3dfee058bcad6dab4af313d9de64a9531666ad72fb`. It establishes the immediate LR `0.05` endpoint benefit in this setup and its persistence after inherited S17 momentum-buffer values are reset. Inherited S17 momentum cannot by itself explain the endpoint benefit; it affects the measured secondary online training-loss trajectory. Deterministic and stochastic finite-step mechanisms remain unresolved. This experiment does not reopen those conclusions and is **not** a definitive D-versus-N separator.

The intervention changes gradient averaging, effective gradient-batch construction, and minibatch-gradient variability together. It also raises each branch's training-example exposure from one to four full traversals. Any causal conclusion is therefore about this entire prospectively specified **m=4 controlled-noise intervention**, not a mathematically pure noise effect or a percentage of the historical LR effect mediated by noise.

## 2. Population, units, and fixed cells

Use all 36 outcome-independent historical S17 states, `base_run_id = 1,…,36`. The independent population unit is the **state**. Select historical future replica **A1 for every state**, before viewing any new outcome. No A2, replacement state, alternate future replica, or outcome-based selection is permitted.

For each state there are four required primary cells:

| Gradient regime | LR 0.10 | LR 0.05 | Origin |
| --- | --- | --- | --- |
| `m=1` ordinary minibatch SGD | historical Reset A1 `L_10R` | historical Reset A1 `L_05R` | 72 existing authoritative cells; read only |
| `m=4` same-parameter gradient averaging | new A1-anchored branch | new A1-anchored branch | 72 new branches |

Both regimes start from the same S17 model parameters and the same defined Reset condition: retain the complete SGD optimizer structure and every momentum-buffer entry, set the numerical contents of **all four** inherited S17 momentum buffers to zero before epoch18, then allow normal within-epoch momentum accumulation. The coefficient stays `0.9`; dampening `0`, weight decay `0`, Nesterov `False`, parameter-group membership and order, model, data/split, loss, evaluation, and other scientific settings stay fixed. The only within-regime LR-pair difference is LR `0.10` versus `0.05`. Each branch is independently instantiated from immutable S17 evidence; one LR branch is never continued from the other.

The new experiment produces `36 × 2 = 72` m=4 branches. Confirmatory analysis requires the 72 existing m=1 A1 cells plus these 72 new cells. Branches and their four component gradients are **not** independent states. No Keep cell participates.

## 3. Historical m=1 A1 admission gate

The sole m=1 source is the Momentum-Reset v1 production directory `internal-artifacts/reflexml-momentum-reset-v1-production-20260927-001`. Its completion manifest is `momentum_reset_completion_manifest.json`, SHA-256 `69260f4fbad9b2db434520949edb2ab1a6464fdc1e0eeaf1d6e554a15b521efa`. This digest identifies the source manifest for this candidate; every branch record must be checked individually at admission. For each `s=1,…,36`, select **only** manifest keys `{s:03d}-A-r01-L_10R` and `{s:03d}-A-r01-L_05R`, and only each key's first scientifically valid authoritative attempt. Resolve its recorded filename within that production directory. Require exact file-byte SHA-256 equal to manifest `file_sha256`, and validate the record's internal `record_sha256` by its original canonical-record rule. Require no open, unresolved, conflicting, substituted, or later preferred attempt.

For both admitted cells, verify `base_run_id=s`, `replica_id=1`, Reset treatment and assigned LR, protocol/source identity, identical exact S17 source-artifact and checkpoint SHA-256, model-parameter identity, preserved optimizer mapping, zero-valued inherited momentum buffers before training, canonical post-S17 generator/future-seed identity, historical A1 epoch18 realized order and batch boundaries, `79` optimizer steps, endpoint definition, and validation dataset/evaluation semantics. Require the two realized A1 orders to match each other exactly, including all 10,000 global example indices and the final 16-example batch, and to match the bound historical A1 expected-order hash. Verify that the record's `val_loss` and `val_accuracy` are the unchanged endpoint values. No compatibility-replay, regenerated historical record, analysis summary, A2 cell, or alternative attempt may substitute.

An independent pre-analysis numerical-compatibility gate must compare the historical m=1 kernel and environment with the proposed m=4 implementation: S17 reconstruction, Reset, data transforms and split, per-example and batch-mean loss, optimizer arithmetic and state, parameter dtype/device, validation function, and endpoint timing. The intended m=4 accumulation is the sole authorized numerical difference beyond extra stream exposure. If identity, integrity, or compatibility cannot be established, stop admission and mark the complete 36-state primary comparison unresolved; do not drop a state, rerun m=1, or choose another record. This protocol does not assert that future compatibility review has already passed.

The historical pooled A1/A2 Momentum-Reset estimate is **not** `Delta_1` here. Recompute the A1-only m=1 contrast from admitted cell values under this protocol's state aggregation and bootstrap, without altering historical frozen analyses.

## 4. Exact four-stream construction

Let `T = (T[0],…,T[9999])` be the single canonical **ordered** Phase 5 `train_indices` sequence, common to all 36 states. `T[i]` is the official Fashion-MNIST training-set **global example index** stored at local position `i` in the original, unsorted `train_indices` used by the authoritative historical loader in `reflexml/data.py` (`make_split_indices` and `make_loaders_from_datasets`, which pass this sequence to `Subset`/`IndexedSubset`). The binding authority is `phase5_preflight_r4/dataset_identity.json`, exact file SHA-256 `18fcff67679cf832d9398aa8251826b57470d5554de9180ace1fda733a923898`, with `train_indices_sha256 = de2fe7734daf05c6820f4906ad4d561661bb952a99549bfcf6c4a61d3bdd113d`, split seed `2026`, and its registered split fingerprint. Before constructing streams 2–4, verify the identity of this registration and historical loader, reconstruct/read the original ordered sequence under the bound historical semantics, and require **element-by-element equality** between `T` and the actual loader's local-position→global-example mapping, as well as the registered ordered-sequence SHA-256, dataset bytes, and transform. Mere set equality is insufficient; do not sort `train_indices`, reconstruct it from an unordered set, or accept a reordered but population-equivalent list. The unique prospective mapping is `local position i → T[i] → global training example`. If this ordered mapping or its authority cannot be verified, the m=4 branch fails admission; no alternate ordering may substitute. The four streams use exactly this 10,000-example set, once each. Record each 10,000-index global order and every batch boundary in an auditable pre-outcome stream manifest; bind its bytes/hash at formal freeze. A hash alone is insufficient if the actual order cannot be recovered.

**Stream 1:** Use the exact historical Reset A1 epoch18 realized global-index ordering and batch boundaries admitted in §3, with no new shuffle. Its equality between historical LR cells is required. The new LR pair receives this same stream. `A1` names the historical anchor; it does **not** imply that streams 2–4 existed in the historical A1 run.

**Streams 2–4:** The following prospective, portable hash-based construction makes three distinct pseudorandom permutations for each state. It uses no Python `hash()` and no NumPy/PyTorch RNG for stream generation.

- Master seed, interpreted as exactly 32 bytes from lowercase hexadecimal: `6a87c34e90f1b52d46e0a9c78d35f12b8e441096c2d7a5ef39b0684d21c59f73`.
- For each `s` in `1..36` and stream index `j` in `2,3,4`, construct the message as the ASCII bytes `ReflexML-Gradient-Noise-Attenuation-v1/stream/` followed by `s` encoded as an unsigned 32-bit **big-endian** integer and `j` encoded as one unsigned byte. There are no separators or newline after the ASCII prefix.
- Let `K[s,j] = HMAC-SHA256(master_seed_bytes, message)`, the full 32 digest bytes. This is the exact stream seed/key derivation. No truncation or mapping to a NumPy/PyTorch seed range occurs because the specified permutation does not use their RNGs; replacing this construction with a library shuffle would be a scientific change requiring prospective review.
- For every local position `i=0..9999`, compute `H[s,j,i] = SHA256(K[s,j] || uint32_be(i))`. Sort all `i` by the lexicographic order of the 32 **raw digest bytes**, breaking an exact digest tie by ascending `i`. The sorted sequence of positions `p_0..p_9999` defines the global-index stream `T[p_0],…,T[p_9999]`.

The three keys are domain-separated by state and stream index, fixed before outcomes, and computationally independent pseudorandom stream constructions; this does not assert independence of scientific outcomes or of the shared training examples. Audit uniqueness of each permutation, equality of its global-index set to the historical A1 stream, and absence of duplicate stream identities. This set check does not replace the required **ordered** `T`-to-loader mapping equality above. If the audited stream identities fail, stop before production; do not select replacement seeds. Both LR branches of a state use byte-identical four stream orders and boundaries. Record exact key digests, ordered-index hashes, source list identity, and the full materializable orders before outcomes.

## 5. Minibatches and optimizer semantics

For each stream, split its 10,000-index ordering contiguously into **79** positions: `t=1..78` have 128 examples per stream; `t=79` has exactly 16 per stream. No dropped or repeated example occurs within a stream. Thus an m=4 branch has four full training-set traversals and **79**, not 316, optimizer steps. At any `t`, the four component batch sizes match: `(128,128,128,128)` for `1..78` and `(16,16,16,16)` for `79`. Never align a short batch with a full batch. The four component losses use the historical mean-over-examples cross-entropy definition. Equal arithmetic weighting of their gradients is valid here **because all four component batch sizes match at each position**; at `t=79` the mean is over four 16-example component gradients, equivalent at fixed parameters to a mean over 64 examples. At earlier positions it corresponds to 512 examples.

At every position `t`, set a single pre-update parameter vector `theta_t`. In model **train** mode, calculate the four separate component gradients `g_1(theta_t),…,g_4(theta_t)` under that unchanged vector and the same loss reduction. Clear or isolate temporary `.grad` buffers so each component gradient is individually well-defined; form the arithmetic mean `g_bar_t=(g_1+g_2+g_3+g_4)/4` for **every** trainable parameter, and place only this mean in the `.grad` buffers used by the optimizer. Take exactly **one** momentum-SGD step after all four gradients have been formed. Clear buffers before the next position. No model parameter, optimizer state, momentum buffer, LR scheduler state, or future scientific RNG state may advance between component gradient evaluations, apart from the prospectively specified data-loading progression that yields these four fixed batches. No gradient clipping, loss rescaling beyond the specified arithmetic average, sequential optimizer steps, or extra validation evaluation is allowed.

The current `FashionMLP` is `Flatten → Linear(784,128) → ReLU → Linear(128,10)` and contains no dropout, batch normalization, or other stochastic/state-mutating forward component. This is a **protocol condition to reverify at implementation audit**, not an assumption that survives code drift. If a stochastic or mutable forward operation appears, the numerical-compatibility gate fails until a separately reviewed protocol revision resolves its handling. Evaluation uses model **eval** mode only after step 79, with no gradient update. Match historical CPU/device, dtype, deterministic settings, numerical loss kernel, and SGD behavior to the extent needed for an interpretable comparison. Store pre-step and realized order/gradient-averaging evidence sufficient to audit fixed-parameter accumulation and the count of optimizer updates.

## 6. Outcomes, estimands, and aggregation

The **sole primary outcome** is epoch18 validation loss **after the 79th optimizer update**, using the same fixed 5,000-example validation split, cross-entropy reduction, evaluation function, and timing as LR-Switch and Momentum-Reset. No official test-set information enters design, execution choice, or analysis. Epoch18 validation accuracy at that same endpoint is prespecified **secondary** and cannot replace or rescue the primary result.

Historical online training loss accumulates losses along one 10,000-example traversal, whereas m=4 would evaluate 40,000 example exposures over four component paths. It is excluded from confirmatory cross-regime secondary inference. It may be recorded only as an operational diagnostic, with no matched scientific endpoint claim.

For each state `s`, let `L[s,m,lr]` be the primary outcome in the admitted A1-anchored cell. Define:

- `d[s,1] = L[s,1,.10] − L[s,1,.05]`
- `d[s,4] = L[s,4,.10] − L[s,4,.05]`
- `Delta_1 = mean_s d[s,1]`
- `Delta_4 = mean_s d[s,4]`
- `Psi_4 = mean_s (d[s,1] − d[s,4]) = Delta_1 − Delta_4`
- `R_.10 = mean_s (L[s,1,.10] − L[s,4,.10])`
- `R_.05 = mean_s (L[s,1,.05] − L[s,4,.05])`.

`Psi_4` is the **primary effect-modification estimand**. Positive LR contrasts mean LR `0.05` has lower validation loss. Positive `Psi_4` means m=4 attenuates that advantage; negative means it amplifies it. Positive `R_lr` means m=4 lowers validation loss at that fixed LR. Require the statewise and aggregate algebraic identity `Psi_4 = R_.10 − R_.05` up to declared floating-point representation tolerance; an unexplained mismatch blocks analysis. Report all five quantities and all four cell means. Compute **one** matched contrast per state and average across exactly 36 states. There is no within-state replica averaging and no branch-level independence assumption. Apply analogous signed four-cell arithmetic to validation accuracy only as secondary, while remembering that *higher* accuracy is better.

## 7. Frozen-intent uncertainty and interpretation

If all 144 required cells pass admission and have finite primary outcomes, use a paired state-level percentile bootstrap with `B=10,000`. The bootstrap seed is the prospective unsigned decimal integer **`202609270418`** supplied to NumPy `Generator(PCG64)`. In replicate order `b=1..10000`, draw exactly 36 independent uniform integers in `[0,36)` using the generator's `integers(low=0, high=36, size=36)` operation; map integer `0` to `base_run_id=1`, …, integer `35` to `base_run_id=36`. Retain all four matched cells for every sampled state occurrence. Use **the same sampled state indices** for `Delta_1`, `Delta_4`, `Psi_4`, `R_.10`, `R_.05`, and secondary accuracy. For each quantity report the point estimate and empirical `q025`/`q975` percentile interval from the 10,000 replicate estimates, using the linear-interpolation empirical quantile convention (`method="linear"` in NumPy). Bind the NumPy/PCG64 implementation version and materialize and hash the entire 10,000 × 36 resampling-index matrix before accessing new outcomes, so later library drift cannot select a different resample. The bootstrap seed, resampling unit, and interval method must be frozen before accessing new outcomes. Do not add an alternative uncertainty method after seeing results.

For the **primary** interval of `Psi_4`:

- Entirely above zero: controlled m=4 gradient averaging **attenuates** the immediate LR `0.05` validation-loss advantage relative to ordinary m=1 SGD. The LR effect is causally sensitive to the prospectively specified m=4 intervention, which attenuates minibatch-gradient variability while also changing effective gradient-batch construction and training-example exposure. This result is compatible with a stochastic-gradient contribution, but does not establish one uniquely.
- Entirely below zero: the intervention **amplifies** the LR `0.05` advantage. Effect modification runs opposite to the simple claim that minibatch noise creates the advantage; a unique mechanism remains unidentified.
- Contains zero, including an endpoint equal to zero: modification is **not established at achieved precision**. This is not equivalence, no effect, proof of deterministic dynamics, or proof that stochasticity is irrelevant. No numerical equivalence/materiality margin is prespecified, and none may be invented after outcomes.

Always report `Delta_4`: an interval entirely above zero means the LR `0.05` advantage **persists under m=4**; only the amount/type of variation attenuated by this specific intervention is not necessary for its existence. An interval containing zero means the m=4 LR advantage is **not established at achieved precision**, not that it vanished or equals zero. An interval entirely below zero means LR preference **reverses under m=4** and demonstrates regime dependence without isolating a unique mechanism. Report `Delta_1` as the A1-only reference under this protocol, without substituting the historical pooled A1/A2 estimate.

Always report `R_.10` and `R_.05`, their intervals, and all four cell means. Use their signs, intervals, and the identity above to locate whether any supported interaction reflects change at LR `0.10`, LR `0.05`, or both; avoid describing an unsupported fixed-LR direction as established. Do not summarize every interaction as “noise removal helped” or “noise removal hurt.” Secondary accuracy is labeled secondary throughout and cannot override the primary loss classification.

## 8. Technical pilot, failures, and analysis admission

A strictly **technical, non-scientific pilot** may precede production. Prefer a non-authoritative fixture. It may check fixed-pre-update parameters, separate component `.grad` buffers, arithmetic averaging, final 16-example batches, permutation and LR-stream identity, deterministic reconstruction, optimizer-step count, memory, and runtime. It may **not** inspect aggregate LR endpoint contrasts, choose `m` or seeds from results, change the scientific design, or enter the confirmatory cells. If a real historical state is used for a pilot, its scientific result is excluded from confirmatory production by default; this protocol does not prospectively authorize its reuse. A fresh, separately authorized production attempt must follow the final audited implementation, and a pilot cannot silently create an alternate record for an existing authoritative cell.

Before any future production, bind the formal protocol bytes/SHA-256, implementation/kernel and environment identities, immutable S17 source and checkpoint hashes, Reset pre-step evidence, four stream orders/hashes, both LR assignments, endpoint semantics, and all expected branch identities. Preserve append-only pre-step and attempt evidence, including technical failures, terminal classification, output hashes, and retry lineage. A complete-set accounting must cover exactly 72 new branches and identify one authoritative first scientifically valid result or an unresolved disposition for each. Do not build a new scheduler or control plane merely to satisfy this evidence requirement.

An attempt is **technical-invalid** only on objective evidence that the specified scientific computation was not performed or cannot be authenticated: wrong or unverifiable S17/checkpoint/Reset/stream/LR/optimizer/implementation identity, an unintended intermediate step or state/RNG advance, missing/truncated/corrupt artifact, demonstrable infrastructure failure, or incompatible numerical kernel. Document the failed condition independently of result magnitude or direction. A retry is allowed only after objective technical-invalid classification and explicit future authority, with exactly the same state, A1 anchor, S17 source/checkpoint, Reset state, LR, four stream orders, optimizer semantics, and protocol; link the prior attempt. If a prior attempt is open or its terminal status uncertain, reconcile it before any retry. No alternate seed, state, stream, code path, or easier record is a retry.

The **first scientifically valid attempt** is authoritative even if its loss is surprising, extreme, NaN, or Inf. A valid numerical divergence is a scientific outcome, never by itself a technical failure. If any required historical m=1 cell cannot pass admission, any new branch is unresolved, or a scientifically valid required primary cell is non-finite, the 36-state primary `Psi_4` analysis is **NON-EVALUABLE AS SPECIFIED**. Preserve all cells and their dispositions; do not silently drop, impute, clip, replace, rerun until finite, or use available-case inference. A scientifically valid non-finite secondary accuracy blocks its own secondary analysis, not a finite primary loss analysis. Any scientific redesign requires a separate prospective protocol and independent review, not an outcome-driven amendment of this protocol.

## 9. Production accounting, compute, and stopping boundary

The fixed prospective roster is 36 states × one A1-anchored m=4 construction × two LRs = **72 new scientific branches**. Each branch has four traversals of the fixed 10,000-example training subset and 79 optimizer steps: **288 ordinary dataset-pass equivalents** for new training. The rejected exact-gradient reference would be approximately **5,688 dataset-pass equivalents**. m=4 is selected prospectively for a lower-cost controlled attenuation study, not from observed m=4 results. This cost comparison does not make the regimes exposure-matched.

After the complete roster and admission gates, perform the fixed primary analysis once, then the prespecified secondary accuracy analysis if evaluable. If the result is weak or inconclusive, report it as such and stop. Do not add `m=2`, `m=8`, exact full-gradient branches, A2 m=4, extra replicas, longer horizons, a large LR grid, new architecture/dataset, Hessian/sharpness work, or confirmatory trajectory visualization in this experiment. Any later question requires a separately specified and authorized study.

## 10. Frozen-intent claims and explicit non-claims

The strongest allowed causal statement concerns sensitivity of the epoch18 LR contrast to **controlled m=4 gradient averaging** in these 36 historical states. A result may be compatible with a stochastic-gradient account, but this intervention cannot isolate its contribution or show stochasticity as a whole absent or uniquely causal. The protocol makes **no** equivalence claim when `CI(Psi_4)` includes zero; no claim that m=4 is zero-noise or a pure variance manipulation; no claim that stochasticity is irrelevant; no proof of deterministic dynamics `D` or unique stochastic mechanism `N`; no exact D/N decomposition or mediation percentage; no Hessian, sharpness, gradient-alignment, or momentum-mechanism claim; no external generality or novelty claim. The prior Momentum-Reset scientific closure remains closed.

## 11. Status and execution prohibition

This protocol is **FROZEN**, following targeted independent re-audit, but is not production authority. Required later gates are: implementation; independent implementation/numerical-compatibility audit including the historical m=1 admission gate; separate explicit production execution authorization; execution and integrity completion; then frozen primary and secondary analysis. A technical pilot, completed audit, or formal freeze does **not** authorize scientific training. **No m=4 production execution is authorized by this document.**
