> Public presentation edition of the authoritative freeze snapshot. Internal numerical authorities are preserved; local paths are adapted. The later [global novelty provenance note](NOVELTY_AND_STOPPING.md) closes the standalone-note gap without supplying a new external audit.

<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# ReflexML final scientific snapshot

## A. Project status

**COMPLETE / FROZEN** — freeze date: **2026-09-29 (Asia/Shanghai)**.

**FREEZE AND WRITE UP AS A CONTROLLED STUDY.** No further scientific experiments are authorized under the current ReflexML research question.

This document is authoritative for the final project status, synthesis, and claim ceiling. Experiment-specific frozen protocols and saved numerical artifacts remain authoritative for definitions, values, and provenance. The [research postmortem](RESEARCH_POSTMORTEM.md) records lessons, not new scientific findings. Historical plans and execution authorizations are retained as history and do not authorize continuation after this freeze.

The final project-level novelty decision supplied in the closing instruction is **NONE — NO DEFENSIBLE PAPER POINT AT ACCEPTABLE MARGINAL COST**. It is a contribution/value judgment, not a conclusion that all hypotheses were wrong.

No Phase 5B salvage, K_A/K_B/N expansion, d5/d6/d7, further MNIST expansion, new datasets/architectures, Hessian/curvature campaigns, gradient-noise fishing, optimizer comparisons, mechanism searches, state-feature mining, predictive-model fishing, post-hoc timing laws, d4 publication-claim manufacture, or relabeling existing results as methodological/causal novelty is authorized. Documentation, verification of existing artifacts, consistency checks, and figures from frozen outputs remain allowed.

## B. Original question and evolution

The original framework compared learning-rate actions from the same checkpoint, with matched optimizer state and stochastic futures, to ask whether training state could inform a useful intervention. Early work showed that simply reducing LR was a weak problem formulation and that a single future could produce unstable timing labels. Phase 5 therefore separated an outcome-independent replication of the average short response (5A) from the state-level association between expected short response rho(S) and expected final timing effect tau(S) (5B).

Phase 5A established a large h1–3 Wait-minus-Now gap that nearly disappeared at h4; Phase 5B did not resolve the association. The h3→h4 attenuation became a follow-up because it constrained what a short response could mean. Exact replay localized historical closure; same-S17 LR branching established immediate endpoint LR causality; Momentum Reset and m=4 averaging constrained particular explanations without identifying a unique mechanism. These follow-ups did not retroactively become Phase 5 primary endpoints.

MNIST screening tested transfer and remained AMBIGUOUS. Wait-d tested whether attenuation was specific to the Wait3 reunification boundary and found closure across tested d=1–4. The global contribution review then rejected continued novelty-seeking: locally informative extensions had not produced a defensible paper point at acceptable marginal cost. The final decision preserves the controlled study and stops experimentation.

## C. Final evidence table

Statuses describe the scoped claim in each row; they are not publication eligibility labels. Native protocol classifications such as GO and AMBIGUOUS remain in the Result column.

| Question | Experiment | Result | Status | Allowed conclusion | Forbidden overclaim | Authoritative artifact |
| --- | --- | --- | --- | --- | --- | --- |
| Does the short response replicate and attenuate at h4? | Phase 5A, 36 states × 15 A futures | Signature supported; negative A34 | ESTABLISHED | A strong h1–3 validation-loss gap nearly disappears at h4 in this setup | Universal collapse law; mechanism identified; exact zero/equivalence | [Design][p5design]; [results][p5results]; [saved JSON][p5json] |
| Does rho associate with tau? | Phase 5B, disjoint A/B blocks, 36 states | psi interval crosses zero | INCONCLUSIVE | Association precision insufficient | No association; negative association established; inherently unpredictable long-term effects | [Results][p5results]; [saved JSON][p5json] |
| Are latent state effects/heterogeneity identified? | Post-outcome EIV and precision review | Weakly resolved signal; rho variance support includes zero | INCONCLUSIVE | Noise and finite-state uncertainty limit state-level interpretation | No state effects exist; positive variance point/shifted bootstrap proves heterogeneity; EIV rescues 5B | [EIV report][eiv]; [precision review][precision] |
| Does the LR transition contribute immediately? | Same-S17 LR .10 versus .05, 36 × A1/A2 | delta_switch CI positive | ESTABLISHED | LR transition directly contributes to validation-loss difference in this controlled comparison | Entire historical collapse explained; universal optimizer law | [LR protocol][lrprotocol]; [primary JSON][lrjson] |
| Is inherited S17 momentum sufficient? | LR × Keep/Reset, 36 × A1/A2 | LR benefit survives Reset | WEAKENED | Inherited S17 momentum alone is insufficient as an endpoint explanation | Momentum irrelevant; all momentum dynamics removed | [Primary JSON][mrjson]; [mechanism closure][mrclosure] |
| Does inherited momentum modify the primary LR effect? | Momentum Reset primary validation loss | Interaction CI crosses zero | INCONCLUSIVE | Primary effect modification unresolved | Zero interaction; equivalence | [Primary JSON][mrjson] |
| Does inherited momentum affect the secondary online-loss outcome? | Momentum Reset secondary analysis | Online-loss interaction CI positive | ESTABLISHED | Outcome-specific effect on losses accumulated along epoch18 | Replace primary endpoint conclusion; identify parameter-path mechanism | [Mechanism closure][mrclosure] |
| Does the LR contrast persist under four-gradient averaging? | Same-parameter m=4, Reset state, 36 × A1 | Delta4 positive; Psi4 positive | ESTABLISHED | Contrast substantially attenuated, not eliminated, under the whole intervention | Pure denoising; proof noise causes the effect; unique mechanism | [m=4 protocol][m4protocol]; [primary JSON][m4json]; [report][m4report] |
| When does historical closure occur? | Exact 36-state A1 historical replay | Front-loaded after initial expansion; Wait improvement dominates whole epoch | ESTABLISHED | Closure is early/front-loaded at frozen probes; final short batch is not the primary explanation | Full future-randomness population trajectory; arithmetic mediation; every early segment driven by Wait | [Historical report][histreport]; [summary][histcsv] |
| When does the controlled LR contrast form? | Exact 36-state A1 LR replay | Immediate component, front-loaded body, non-monotone observed path, material final increment | ESTABLISHED | Broad timing localized for this A1 sample | Identical shape to historical closure; independent replication; unique causal explanation of closure | [LR dynamics report][dynreport]; [table][dyncsv] |
| Does strong collapse transfer to MNIST? | External v0.1, N=6→12, K=2 | AMBIGUOUS; partial attenuation | PARTIAL | Strong collapse did not replicate; closure strength differs across the two tested experimental settings | Successful strong replication; dataset identity causes the difference | [N=6 screen][mn6]; [N=12 analysis][mn12]; [MNIST implementation][mncode] |
| Is attenuation specific to Wait3? | Wait-d v0.1, Fashion-MNIST N=12, K=2, d=1–4 | GO; BROADLY_SIMILAR_CLOSURE | ESTABLISHED | Reunification-linked attenuation is not specific to Wait3 over the tested grid in this setting | GO means continue research; monotonic law; relaxation constant; d4 novelty | [Analysis][wdanalysis]; [validation][wdvalidation]; [manifest][wdmanifest]; [implementation][wdcode] |
| Is a unique mechanism established? | LR, Reset, m=4 and trajectory constraints together | Particular explanations weakened; alternatives unresolved | INCONCLUSIVE | Useful controlled constraints without mechanism identification | Deterministic/noise/momentum/curvature mechanism proven; causal mediation fraction | [Momentum closure][mrclosure]; [m=4 report][m4report]; [trajectory reports][dynreport] |
| Is generic LR-transition reconvergence/contraction new? | Existing literature audit | Strong near-neighbor prior art | PRIOR-ART-COVERED | A case-level controlled study with explicit limits | Exact lexical mismatch establishes novelty; rigorous branching is a novel method | [Retained narrow audit][novelty]; final closing decision in A/G |

## D. Frozen numerical results

Values below are copied from saved artifacts at their recorded precision; no analysis, bootstrap, evaluation, or training was rerun. Loss contrasts are absolute validation cross-entropy differences unless explicitly normalized. A positive Wait−Now or .10−.05 contrast favors Now or LR .05, respectively.

### Phase 5A and 5B

Fashion-MNIST; N=36 outcome-independent epoch14 states; K_A=15 and K_B=10 disjoint futures. Now uses .05 from epoch15; Wait3 uses .10 in epochs15–17 and .05 from epoch18; both finish at epoch30. h1–4 are epochs15–18. rho is the conditional mean of (Delta1+Delta2+Delta3)/3. tau is the conditional mean of **per-replica** relative gains, each using branch mean losses over epochs28–30: (L_W−L_N)/max(L_W,1e−12). psi is the across-state covariance, estimated with denominator N−1, not a correlation.

Source: [saved primary JSON][p5json]. Intervals: frozen 20,000 nested whole-state/whole-replica PCG64 percentile draws, seed 10698172564239836591.

| Estimand | Saved point estimate | Saved 95% interval |
| --- | ---: | --- |
| `mu_rho` | `0.04846931827645519` | `[0.046131427886917806, 0.050900150152920716]` |
| `deltaL1` | `0.04815040450080677` | `[0.04433033686385525, 0.05223866010203198]` |
| `deltaL2` | `0.050203690413225574` | `[0.046043779926317856, 0.05445547198015107]` |
| `deltaL3` | `0.0470538599153332` | `[0.04351884824952587, 0.050783662359226496]` |
| `deltaL4` | `9.025069421640319e-05` | `[-0.0013578638559040357, 0.0015151950749506564]` |
| `A34` | `-0.04696360922111681` | `[-0.050983711435128067, -0.04307340872724224]` |
| `mean_final_effect` | `0.0002677283129176079` | `[-0.0021670669561728154, 0.002663347596706996]` |
| `psi` | `-6.950007438972617e-06` | `[-2.5196934024589843e-05, 1.0202374944146862e-05]` |

The raw tau between-state variance component is `9.107464511192588e-06`. Its saved nested-bootstrap interval `[9.766482795548284e-06, 6.302776701027449e-05]` has a documented upward shift and must not be read as calibrated proof of positive latent heterogeneity. [Results][p5results] explain the verification.

### EIV / measurement precision — post-outcome exploratory

Source: [saved decomposition][eivcsv].

| Quantity | rho | tau |
| --- | ---: | ---: |
| Observed across-state variance | `2.7077867740521444e-05` | `3.367929005776853e-05` |
| Mean state-estimator measurement variance | `2.860088218563167e-05` | `2.457182554657594e-05` |
| Raw deconvolved variance | `-1.5230144451102272e-06` | `9.107464511192588e-06` |
| Raw signal fraction | `-0.05624573026594221` | `0.2704173542723429` |

The negative rho component means positive heterogeneity is unresolved, not negative physical variance or exact homogeneity. The tau fraction is a reliability-style estimate, not known reliability. EIV support sets are descriptive likelihood sensitivity, not calibrated 95% CIs; rho variance includes zero and positive-variance latent-correlation profiles cover [-1,1], with correlation undefined at a zero-variance boundary. No corrected-correlation claim replaces 5B.

### LR-switch endpoint

Same historical pre-switch S17 model, optimizer and future stream; N=36, A1/A2 (K=2). `delta_switch=L_Continue18−L_Switch18`; `D=L17−L_Switch18`; `kappa_C=L17−L_Continue18`. D=kappa_C+delta_switch is arithmetic, not mediation. Source: [primary JSON][lrjson]; 10,000 state draws, PCG64 seed 12114954806208443255.

| Estimand | Saved point estimate | Saved 95% interval |
| --- | ---: | --- |
| `delta_switch` | `0.04110032269702188` | `[0.033847031257023194, 0.04871296943619432]` |
| `D` | `0.04584928999013371` | `[0.036812896945766276, 0.05627296271870325]` |
| `kappa_C` | `0.004748967293111818` | `[-0.007571413025207204, 0.016896603213218557]` |

Continue catch-up remains unresolved; LR is not shown necessary for all catch-up.

### Momentum Reset

N=36, A1/A2; epoch18 validation loss; all four inherited S17 buffers reset jointly before training, with epoch18 momentum dynamics retained. `Delta_K=L_.10,Keep−L_.05,Keep`; `Delta_R=L_.10,Reset−L_.05,Reset`; `I=Delta_K−Delta_R`. Source: [primary JSON][mrjson]; 10,000 shared state draws, seed 6630267477959695398.

| Estimand | Saved point estimate | Saved 95% interval |
| --- | ---: | --- |
| `Delta_K` | `0.04110032269702188` | `[0.033806524633044996, 0.048786007891713314]` |
| `Delta_R` | `0.044404765195026986` | `[0.03713075117724119, 0.052416257122717774]` |
| `I` | `-0.003304442498005098` | `[-0.008549983271135838, 0.0018922622160199621]` |

The Keep point is identical to LR-switch, but its interval differs because this factorial analysis has its own frozen draw matrix. Secondary online-loss interaction is reported as `0.006900180`, CI `[0.004886686, 0.009054503]` at the precision of the [closure report][mrclosure]; it does not replace the validation-loss endpoint.

### Same-parameter four-gradient averaging

N=36, A1 only; inherited momentum Reset in both m=1 and m=4. Four gradients are evaluated at the same parameters and averaged for each update. This changes batch construction/exposure as well as variability; m=4 is a composite intervention. `Delta_m=L_m,.10−L_m,.05`; `Psi4=Delta1−Delta4`. Source: [primary JSON][m4json]; 10,000 shared state draws, seed 202609270418.

| Estimand | Saved point estimate | Saved 95% interval |
| --- | ---: | --- |
| `Delta_1` | `0.04841821200044619` | `[0.03621787444597938, 0.062439635780919336]` |
| `Delta_4` | `0.011728252187122902` | `[0.009196234473362567, 0.014518654011483823]` |
| `Psi_4` | `0.036689959813323285` | `[0.024364725643052823, 0.05067324152987657]` |

### Historical collapse and LR-switch dynamics

Both use 36 states × historical A1 only, epoch18, 79 updates, probes {0,1,3,8,16,32,48,64,78,79}; 72/72 endpoint gates PASS in each saved report. State-only 20,000 PCG64 draws retain paired arms/all probes; intervals are pointwise, not simultaneous. Neither analysis integrates over the future-randomness population.

Historical `C_t=mean(Wait3−Now)`; `F_t=(C0−Ct)/(C0−C79)`, descriptive and untruncated. Source: [saved aggregate][histcsv].

| Update t | C_t | F_t |
| --- | ---: | ---: |
| 0 | `0.052862266864544816` | `0.0` |
| 1 | `0.05628002818127473` | `-0.06377291067874523` |
| 8 | `0.022847827126913602` | `0.5600473547087663` |
| 16 | `0.011317949093050431` | `0.7751863927659665` |
| 32 | `0.0027028726778096615` | `0.9359373779299579` |
| 79 | `-0.0007304150493608531` | `1.0` |

Whole-epoch Wait catch-up accounts descriptively for 87.674665% of closure; Now deterioration for 12.325335%, at [report precision][histreport]. The early t0→t3 segment is primarily Now deterioration; whole-epoch dominance must not be imposed on every segment. The final t78→79 closure increment is `0.001520173572`, CI `[-0.000589063695, 0.004106318015]`, 2.836532% of final closure (report precision). Sparse probes cannot identify an exact collapse step.

LR `D_t=mean(L_.10−L_.05)`; `G_t=D_t/D79`, descriptive. Source: [saved primary table][dyncsv].

| Update t | D_t | Pointwise 95% CI | G_t |
| --- | ---: | --- | ---: |
| 0 | `0.0` | `[0.0, 0.0]` | `0.0` |
| 1 | `0.020119659915732016` | `[0.011583179489187898, 0.02949057958194364]` | `0.44971419292728243` |
| 8 | `0.03508144957290756` | `[0.020952610845009066, 0.05281760520483882]` | `0.7841397840459113` |
| 79 | `0.04473877016148634` | `[0.03353113795293909, 0.05607662926329714]` | `1.0` |

The LR final increment is `0.014291442884`, CI `[0.005958286263, 0.023164936010]`, 31.944201% of endpoint contrast (report precision). It differs from historical closure. The trajectories share the Switch/Wait3 arm, so alignment is not independent replication or mediation. Their final-step percentages have different denominators and are not an interaction test. A1 D79 differs from the A1/A2 endpoint delta_switch because replica coverage differs.

### MNIST external screen

N=6 first, protocol-permitted expansion to N=12; K=2. Final **AMBIGUOUS**. h1–4 are epochs15–18 after epoch14 branching. E=(m1+m2+m3)/3; A34=m4−m3. Source: [N=12 JSON][mn12]. No new confidence interval is supplied or inferred.

| Quantity | Saved value |
| --- | ---: |
| `m1` | `0.002159676121054386` |
| `m2` | `0.0031424884370464486` |
| `m3` | `0.002849481517550096` |
| `m4` | `0.0017288039104722253` |
| `E` | `0.0027172153585503104` |
| `A34` | `-0.0011206776070778709` |

Derived descriptive ratios m4/E≈0.636 and m4/m3≈0.607 are rounded summaries, not additional frozen estimands. Partial structural attenuation was observed, but strong collapse did not replicate. Only closure strength across the two tested experimental settings may be compared; dataset identity is not isolated as its cause.

### Wait-d v0.1

Fashion-MNIST; N=12 (historical states 1–12), K=2 (A1/A2), d={1,2,3,4}; all arms rerun from exact epoch14 checkpoints. Now uses .05 from epoch15. Wait-d uses .10 for d epochs, then .05 from epoch15+d; P is the state/future mean gap at epoch14+d and Q at epoch15+d. C=1−Q/P; movement counts `(P_i>0) and (abs(Q_i)<abs(P_i))`. The classifier's per-delay `n` field is this movement count, not a reduced sample size. Sources: [analysis][wdanalysis], [validation][wdvalidation], [bound schedules/code][wdmanifest].

| d | P | Q | C | Movement |
| --- | ---: | ---: | ---: | ---: |
| 1 | `0.04285781789198517` | `0.00041519224941730565` | `0.9903123334355539` | 12/12 |
| 2 | `0.04760556508104006` | `-0.0001571424879133615` | `1.0033009268484863` | 12/12 |
| 3 | `0.054917493730535116` | `0.00265613933329781` | `0.951634003067752` | 12/12 |
| 4 | `0.03844765653051436` | `0.006888789500916995` | `0.8208268039574885` | 10/12 |

Historical Now and Wait3 reproduction gates: **PASS**. Frozen classification **GO**; secondary **BROADLY_SIMILAR_CLOSURE**. GO is the screen's result label, not authority to run more experiments. d4 remains descriptive; no monotonic decline, timing law, relaxation constant, phase transition or asymptotic residual follows.

## E. Negative / inconclusive results

Phase 5B has insufficient precision; neither sign nor absence of association is established, and meaningful state heterogeneity is not reliably identified. Measurement-error sensitivity does not rescue surrogacy or establish that noise is the sole explanation. The final **DO NOT EXPAND K_A** decision is a value/stop decision, not a statistical refutation.

Momentum Reset gives no equivalence conclusion and does not establish the primary interaction. Its positive secondary online-loss interaction prevents a blanket “momentum irrelevant” account. The LR benefit surviving Reset weakens inherited S17 momentum as a sufficient endpoint explanation.

m=4 substantially attenuates but does not eliminate the contrast; its compound nature prevents unique noise attribution. Historical late-only/final-short-batch explanations are weakened by early closure, while LR timing does have a material last-step contribution. Temporal mismatch and shared-arm alignment do not identify a unique alternative mechanism. MNIST remains AMBIGUOUS, with no successful strong-collapse replication. Deterministic, stochastic and interaction explanations remain unresolved.

## F. Claim ceiling

### What ReflexML supports

A rigorously controlled empirical study of LR timing in the specified Fashion-MNIST MLP/SGDM setup: a replicated average short-response signature; near disappearance at h4; controlled immediate LR causality; survival after inherited-momentum Reset; sensitivity to the whole m=4 regime; exact historical A1 temporal localization; bounded Wait-d extension; and an ambiguous external MNIST screen. Observation, intervention effect, mechanism constraint, and generality are distinct evidence levels.

### What ReflexML does not support

A validated predictor/controller or causal surrogate; reliable state-heterogeneity identification; an established negative/zero rho–tau association; proof of long-term unpredictability; a unique momentum/noise/curvature mechanism; causal mediation percentages; equivalence from crossing-zero intervals; universal collapse or timing laws; transfer across architectures/optimizers; successful strong MNIST replication; dataset-causal explanation; independent replication from shared-arm curves; or a standalone methodological/causal novelty claim. The exact local curve or d4 residual cannot be promoted into a publication contribution.

## G. Novelty status

Final governing verdict: **NONE — NO DEFENSIBLE PAPER POINT AT ACCEPTABLE MARGINAL COST**. Generic LR-transition reconvergence/contraction has strong prior art. Wait-d is a useful controlled extension with insufficient standalone novelty. State-conditioned heterogeneity/persistence remains unresolved and would require disproportionate additional experimentation; rho→tau precision expansion is not justified for this project. Same-state paired-future branching strengthens internal validity but is not a standalone novel methodology. Reset plus m=4 constrains mechanisms without establishing a new one.

The retained [2026-09-28 narrow audit][novelty] supports broad prior-art coverage but predates the final MNIST/Wait-d/global stop synthesis. Its moderate-candidate wording and external-replication recommendation are superseded as project decisions. **Source limitation:** no separate final global-audit artifact was located in the inspected repository and identified evidence directories. The exact global verdict and current stop authority are supplied by the user's final-freeze instruction, transcribed here; they are not falsely attributed to the older audit. No new literature search was performed and novelty is not reopened.

Retain ReflexML as **a controlled empirical study / undergraduate research project / reproducibility and research-methods portfolio**.

## H. Why the project stopped

### Scientific stopping reasons

Marginal scientific novelty and expected value no longer justified additional experiments. Strong near-neighbor work covers the broad transition/catch-up phenomenon. The controlled extensions sharpen local knowledge without independently raising the contribution ceiling. State-level association and heterogeneity remain unresolved; external screening did not establish strong transfer; the mechanism discriminators exclude no unique general account. These are reasons to preserve bounded conclusions, not to rescue them with further fishing.

### Engineering / resource reasons

Exact branching/replay, provenance recovery, measurement replication and analysis require engineering time and compute. Those costs matter only relative to expected scientific value. The stop decision is not a claim that compute was unavailable. A statistically useful precision purchase, such as the historical K_A recommendation, can still have insufficient project-level contribution value.

## I. Reusable assets

Reusable assets include full-state checkpoint branching (`reflexml/checkpoint.py`, `reflexml/branching.py`), paired-future seed/stream designs, explicit estimands and state-level analysis utilities, exact replay/order/endpoint checks, frozen protocols, and reporting patterns that distinguish primary, secondary, exploratory, negative and non-evaluable outcomes. These are practical research assets, not claimed novel methods. Retain existing paths and records; do not revive historical orchestration by default. Analysis-only reproduction instructions (internal archive; excluded) remain available; retained training commands describe historical workflows and grant no new experimental authorization.

## J. Authority, provenance and consistency audit

### Source hierarchy and location

- Phase 5: [PHASE5_DESIGN.md][p5design] for protocol; [RESULTS_PHASE5.md][p5results] and [primary JSON][p5json] for results/5B; DECISIONS.md (internal archive; excluded) for decision history. Correct ledger: `internal-artifacts/branch_attempts.jsonl`. The saved JSON's `source.ledger_path` shadowing error is already documented; do not rewrite it.
- EIV/precision: POST_PHASE5_EIV_PLAN.md (internal archive; excluded), protocol/input freeze (internal archive; excluded), [formal report][eiv], [formal manifest][eivmanifest], and [2026-09-28 precision review][precision]. The review's expansion recommendation is historical and superseded; its measurements remain valid at their stated limits.
- LR-switch: [protocol][lrprotocol], [primary JSON][lrjson] and the Stage-I gate/Stage-II completion manifest bound by that JSON. Momentum: [protocol](../MOMENTUM_RESET_PROTOCOL.md), [freeze record](../MOMENTUM_RESET_PROTOCOL_FREEZE_RECORD.md), [primary JSON][mrjson], [closure][mrclosure] and its bound production manifest.
- m=4: [protocol][m4protocol], [freeze record](../GRADIENT_NOISE_ATTENUATION_PROTOCOL_FREEZE_RECORD.md), m=1 A1 admission and stream manifest, [primary JSON][m4json]/[report][m4report]. Preserve the report's procedural deviation: one m=4 endpoint was previewed before common bootstrap-matrix materialization. Seed, draws and estimands were unchanged; full procedural compliance is not claimed.
- Historical collapse: [final scientific report][histreport], aggregate CSV, `analysis_manifest.json`, `integrity_check.json`, `report_finalization_manifest.json`, `artifact_hashes.json` in that analysis directory. LR dynamics: [protocol](../LR_SWITCH_DYNAMICS_PROTOCOL.md), [final scientific report][dynreport] and corresponding manifests/hash inventories; the older implementation handoff (internal archive; excluded) is not the current execution status.
- MNIST and Wait-d: local saved manifests/analyses and [MNIST][mncode]/[Wait-d][wdcode] implementation definitions. No standalone Markdown protocol was located for these screens; saved manifests bind their executed setup. N=12 MNIST incorporates the original N=6 states and frozen epsilon; it is not 12 additional states.
- Novelty: retained [narrow audit][novelty] plus the source-qualified final decision in G. Root README (internal archive; excluded), current navigation (internal archive; excluded), and `.context/research_index.json` are summaries, not numerical authority.

### Checks and resolution

This freeze checked sign, N, K_A/K_B, epoch schedule, d-grid, result classifications and claim boundaries against the sources above. Direct checks found no mismatch in the Phase 5/EIV protocol and input identities, LR-switch/Reset bound gate/completion manifests, the m=4 primary JSON identity, the 90 historical-collapse and 94 LR-dynamics saved analysis-file hashes, or the 121 Wait-d record/manifest bindings. The Wait-d manifest's 13 code-file hashes also match current files. All 49 selected protected source files were unchanged across this documentation work. Local links and principal saved numeric literals were checked. It reused saved execution/replay integrity reports; it did not re-audit every raw production tensor/artifact, rerun analysis, or establish a portable archive.

No principal numerical transcription inconsistency was found. Rounded prompt/report values were replaced only in this new synthesis by the more precise saved values. Apparent discrepancies are scope differences: Phase 5 A-all-futures versus A1 trajectories; A1/A2 LR endpoint versus A1 dynamics; and different frozen bootstrap seeds for Keep intervals. Wait-d d4 `n=10` is movement count, with N still 12. Historical next-work banners, precision expansion and earlier novelty recommendations are superseded by this final decision, without rewriting those frozen records.

Remaining provenance limits: the separate final global-audit source is not present in the inspected locations; many authoritative outputs live outside Git at absolute local paths; current Git contains pre-existing modified/untracked work. **Scientific completion is not Git cleanliness or a self-contained archival guarantee.** Historical records and numerical conclusions remain unchanged.

[p5design]: ../PHASE5_DESIGN.md
[p5results]: ../RESULTS_PHASE5.md
[p5json]: ../results/summaries/phase5_primary_results.json
[eiv]: evidence/EIV_RESULTS.md
[eivcsv]: ../results/summaries/eiv_decomposition.csv
[eivmanifest]: ../results/summaries/eiv_manifest.json
[precision]: evidence/PHASE5B_PRECISION_REVIEW.md
[lrprotocol]: ../MECHANISM_LR_SWITCH_PROTOCOL.md
[lrjson]: ../results/summaries/lr_switch.json
[mrjson]: ../results/summaries/momentum_reset.json
[mrclosure]: ../MOMENTUM_RESET_MECHANISM_CLOSURE.md
[m4protocol]: ../GRADIENT_NOISE_ATTENUATION_PROTOCOL.md
[m4json]: ../results/summaries/m4.json
[m4report]: evidence/M4_REPORT.md
[histreport]: evidence/HISTORICAL_DYNAMICS.md
[histcsv]: ../results/summaries/historical_dynamics.csv
[dynreport]: evidence/LR_DYNAMICS.md
[dyncsv]: ../results/summaries/lr_dynamics.csv
[mn6]: ../results/summaries/mnist_n6.json
[mn12]: ../results/summaries/mnist_n12.json
[mncode]: ../reflexml/collapse_external.py
[wdanalysis]: ../results/summaries/wait_d.json
[wdvalidation]: ../results/summaries/wait_d_validation.json
[wdmanifest]: ../results/summaries/wait_d_manifest.json
[wdcode]: ../reflexml/wait_d.py
[novelty]: evidence/PRIOR_ART_2026-09-28.md