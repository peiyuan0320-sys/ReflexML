<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# Phase 5 primary results — audited freeze

Status as of 2026-09-25 (Asia/Shanghai): `PHASE5_EXECUTION_AUDIT_PASS`; `PHASE5_PRIMARY_ANALYSIS_COMPLETE`. This record freezes the supplied execution-audit verdict and the existing primary analysis; it is not a new analysis. Repository baseline before this documentation freeze: `b72e787cffb78a91fabcba9276a7aa8ef61505b1`. The final ledger event is version 2700 with head `3222eb22d21b2b0bb81aa9062e5444be8c038ffacdf4000dd0639504f45639ab`.

## Sources and identity

- Frozen design: [PHASE5_DESIGN.md](PHASE5_DESIGN.md), SHA-256 `8eff292a2e5027bec893d6cf43e327f961ae91bf390d9ed28a65a996a0799352`.
- Primary branch ledger: `internal-artifacts/branch_attempts.jsonl`, SHA-256 `9382e1a737a96024ffb77d73344a309dc8e540b71491f2d4062e690962fcc634` at freeze inspection.
- Original primary analysis: `internal-artifacts/reflexml-phase5-analysis-v1`; script `analyze_phase5.py` SHA-256 `6b55c4ba94773af7a2c18249b4fae64246ea009e8f198bebda64330422f02ef2`; `phase5_primary_results.json` SHA-256 `77188203f37c30489090c3dbdf08aa81825dd89056ce9ea5b1df5a2d690da416`.

Other saved output SHA-256 hashes at freeze inspection:

| File in primary analysis directory | SHA-256 |
| --- | --- |
| `phase5_state_estimates.csv` | `d119d62176a780fe683ea7b38c630a21a4f0121e350420e7372fd536b1f0a056` |
| `phase5_replica_estimates.csv` | `1868b85cf9ac9f8ac6c158a8af7fa057f92a46ad492bf6a30942b098cace34d1` |
| `phase5_bootstrap_draws.npz` | `be18c4493491858feaa3e3643e736d16fff8c996344a4672b89b9851f627a251` |
| `phase5_horizon_response.png` | `94aea1a43fc2130e0990c956c3ebc94cd5101d6366be7149be34c9638fcddfc4` |
| `phase5_rho_tau_scatter.png` | `211e7882d96a4f05c6659d4bbd7cfbe4cb4e7da3cbe0b9759e3c620db4c7d2d8` |

The command and output boundary are in REPRODUCE_PHASE5.md (internal archive; excluded).

The supplied audit reports 36/36 complete states, 15 A plus 10 B pairs per state, 900/900 successful scientific pairs (540 A, 360 B), 2700 ledger events, 900/900 verified artifacts, zero technical failures, retries, open attempts, and artifact SHA mismatches. This freeze inspection directly checked the ledger file hash and final event/head and read the saved primary JSON; it did not rerun the execution audit or the analysis.

## Frozen design and estimands

The 36 primary epoch-14 states were selected independently of future outcomes. At checkpoint `t*=14`, each state has `K_A=15` short-horizon pairs and `K_B=10` final-effect pairs with disjoint future-replica identities. `ΔL_h = L_W(14+h) − L_N(14+h)` (Wait minus Now). A-block trajectories alone estimate `ρ(S) = E[(ΔL_1+ΔL_2+ΔL_3)/3 | S]`. B-block replicas alone estimate `τ(S) = E[G | S]`, where each replica's `G` uses the two branches' mean validation losses over epochs 28–30: `(L_W−L_N)/max(L_W, 1e−12)`. The preregistered Phase 5B primary association is the across-state covariance of `ρ` and `τ`, estimated with the sample covariance of 36 aligned state estimates. All intervals below are the frozen two-sided 95% percentile intervals from 20,000 nested state-level bootstrap draws using NumPy `Generator(PCG64)` and seed `10698172564239836591`.

## Phase 5A primary results

| Quantity | Point estimate | 95% percentile interval |
| --- | ---: | ---: |
| `mu_rho` | `0.0484693` | `[0.0461314, 0.0509002]` |
| `E[ΔL_1]` | `0.0481504` | `[0.0443303, 0.0522387]` |
| `E[ΔL_2]` | `0.0502037` | `[0.0460438, 0.0544555]` |
| `E[ΔL_3]` | `0.0470539` | `[0.0435188, 0.0507837]` |
| `E[ΔL_4]` | `0.0000903` | `[-0.0013579, 0.0015152]` |
| `A34 = E[ΔL_4 − ΔL_3]` | `-0.0469636` | `[-0.0509837, -0.0430734]` |

Frozen mechanical classification: **`5A signature supported`**. The short response is reproducible over `h=1–3` and sharply attenuates by `h=4` in this design. This does not establish a universal deep-learning law or identify the attenuation mechanism.

## Phase 5B primary results

| Quantity | Point estimate | 95% percentile interval |
| --- | ---: | ---: |
| Mean final effect | `0.0002677` | `[-0.0021671, 0.0026633]` |
| Primary sample covariance `Cov(ρ, τ)` | `-6.9500e-06` | `[-2.5197e-05, 1.0202e-05]` |
| Pearson correlation (companion description) | `-0.230142` | Not a primary interval or claim |

Frozen primary conclusion: **`5B primary association insufficiently precise`** — insufficiently precise evidence of the preregistered linear state-level association at the achieved precision. The point estimates do not prove no association, a negative association, or a validated negative correlation. Pearson correlation is descriptive; the covariance interval controls the primary classification.

## Variance bootstrap verification and provenance note

The saved raw deconvolved between-state variance estimate is `9.107464511192588e-06`, computed as across-state sample variance `3.367929005776853e-05` minus mean within-state B variance `2.457182554657594e-04` divided by 10 (`2.457182554657594e-05`). Its frozen percentile interval is `[9.766482795548284e-06, 6.302776701027449e-05]`. A narrow later check returned `EXPECTED_PERCENTILE_BOOTSTRAP_BEHAVIOR`: point and bootstrap estimators use the same frozen formula; the specified nested bootstrap distribution shifts upward. Do not read this interval naively as strong calibrated evidence that latent between-state variance is bounded away from zero. The check did not alter 5A, the primary covariance, Pearson correlation, or the 5B classification.

The saved JSON's `source.ledger_path` names the last artifact, due to local-variable shadowing in the original loader. It is a provenance-field error, not an input to scientific calculations. The correct primary ledger path is above. The historical JSON and script remain unchanged.

## Non-claims and later work

Phase 5 does not establish universality across datasets, models, or optimizers; the mechanism of the `h=3→4` attenuation; absence of a latent `ρ–τ` relationship; a validated negative `ρ–τ` association; that short response is a reliable long-run decision signal; or external generalization.

Latent/error-in-variables (EIV) sensitivity, secondary checkpoint epochs, fixed-relative-horizon sensitivity, mechanism studies, alternative state features, new models/datasets, and external replication are **not primary results**. They are pending or not yet performed in this freeze record. See DECISIONS.md (internal archive; excluded) before any subsequent work.
