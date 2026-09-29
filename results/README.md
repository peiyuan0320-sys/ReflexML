# Frozen results

Selected JSON/CSV files are reused from saved outputs, without new inference. Numerical values, estimands, intervals, bootstrap seeds and procedural disclosures are preserved. Local absolute paths are replaced by nonportable internal-artifact placeholders. Embedded historical hashes refer to original inputs, not to these adapted copies. SOURCE_MANIFEST.json records both source and public hashes. Final snapshot tables are transcription of those frozen outputs.

- phase5_primary_results.json plus state/replica CSVs: 36 states, disjoint A/B blocks; 5A supported, 5B insufficiently precise.
- lr_switch.json, momentum_reset.json, m4.json: endpoint contrasts and mechanism limits.
- historical_dynamics.csv, lr_dynamics.csv: A1-only fixed-probe trajectories; pointwise intervals.
- mnist_n6.json, mnist_n12.json: screen and expansion, final AMBIGUOUS; N=12 includes the first six states.
- wait_d.json, wait_d_validation.json, wait_d_manifest.json: tested d=1–4, N=12, K=2; GO is a frozen classification.
- eiv_decomposition.csv, eiv_manifest.json: explicitly post-outcome exploratory sensitivity; cannot rescue Phase 5B.

## Figures

1. phase5_horizons.png: Wait3−Now validation loss, N=36, K_A=15; saved pointwise 95% intervals. Positive favors Now.
2. historical_closure.png: historical Wait3−Now validation loss at fixed epoch18 update probes, 36 A1 futures; saved pointwise intervals. Initial expansion precedes front-loaded closure; sparse probes do not identify an exact step.
3. wait_d_closure.png: mean P (last pre-reunification epoch) and Q (first post-reunification epoch), Fashion-MNIST N=12, K=2. Positive favors Now; no confidence intervals were supplied for this plot. The d-grid is descriptive, not a timing law.

No decorative or new inferential figures are included. The MNIST ambiguity and Phase 5B uncertainty are retained in the README and exact snapshot even though they are not additional figures.
