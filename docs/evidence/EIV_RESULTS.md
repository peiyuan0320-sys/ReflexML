<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# Post-Phase-5 EIV results

**post-outcome exploratory**

The frozen Phase 5A and 5B results in `RESULTS_PHASE5.md` are unchanged.

## Distribution-light decomposition

**post-outcome exploratory**

Raw negative variance components mean positive heterogeneity was unresolved by this estimator, not negative physical variance or exact homogeneity.

Outer-only percentiles are sensitivity ranges, not confidence intervals. Their resampling retains replica noise already embedded in the observed means; latent-variance coverage is unestablished.

Outer-only bootstrap: 20000/20000 valid; 0 failed attempts (IDs and failure details in `post_phase5_eiv_fit.json`).
- rho: observed variance 2.70778677e-05; mean measurement variance 2.86008822e-05; raw deconvolved variance -1.52301445e-06; raw signal fraction -0.0562457303; outer-only q025/q500/q975 -1.22777114e-05/-2.42080426e-06/8.68114472e-06.
- tau: observed variance 3.36792901e-05; mean measurement variance 2.45718255e-05; raw deconvolved variance 9.10746451e-06; raw signal fraction 0.270417354; outer-only q025/q500/q975 -5.05646425e-06/7.68866345e-06/2.31621751e-05.
## Conditional plug-in EIV

**post-outcome exploratory**

Full-data unrestricted fit status: `PASS`.
Profile-reference status: `PASS`.

95%-reference likelihood support sets are descriptive. Boundary calibration and nominal 95% frequentist coverage are not established.

If a marginal latent-variance support set includes zero, latent correlation is weakly identified and undefined at the zero-variance boundary.

Conditional plug-in estimates: sigma_rho2=5.37399757e-06, sigma_tau2=1.14838278e-05, psi=-2.83179177e-06; latent correlation: weakly identified; undefined at the zero-variance boundary.
sigma_rho2 95%-reference support: [0, 2.19116043e-05] (exact_zero_boundary, bracketed; resolved); grid edges: {'lower': 'exact_zero_boundary', 'upper': 'outside_support'}.
sigma_tau2 95%-reference support: [1.15844506e-06, 3.18626352e-05] (bracketed, bracketed; resolved); grid edges: {'lower': 'outside_support', 'upper': 'outside_support'}.
psi 95%-reference support: [-1.53605448e-05, 8.17728703e-06] (bracketed, bracketed; resolved); grid edges: {'lower': 'outside_support', 'upper': 'outside_support'}.
r_latent 95%-reference support: [-1, 1] (mathematical_domain_boundary, mathematical_domain_boundary; resolved); grid edges: {'lower': 'mathematical_domain_boundary', 'upper': 'mathematical_domain_boundary'}.
## Measurement variance and robustness

**post-outcome exploratory**

Hierarchical-bootstrap status: conditional on successful fits.

Successful-fit percentiles are conditional on successful fits and make no calibrated coverage claim.

Hierarchical bootstrap: 2000/2000 successful; 0 optimization failures; 0 unstable; 9 boundary fits; undefined-correlation fraction 0.0.
## Precision planning

**post-outcome exploratory**

Median width ranks this planning metric only; where empirical coverage differs, a narrower interval does not establish superior interval performance.

Incomplete cells do not enter Pareto ranking. Frontier position does not select a Phase 6 route.

Simulation cells: 2736.

Complete simulation cells: 2736/2736; non-dominated complete cells: 1056.
See the CSV and JSON artifacts for every cell, failure, and unresolved support point.
