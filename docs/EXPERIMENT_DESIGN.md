# Experimental design

This is a completed study; these definitions describe frozen work and authorize no new experiments.

Phase 5 uses Fashion-MNIST, a fixed MLP and SGD with momentum, 36 outcome-independent epoch-14 checkpoints. From each full checkpoint, Now reduces LR from .10 to .05 at epoch15; Wait3 delays until epoch18. Within a pair, future minibatch order is matched. A has 15 futures per state through epoch18; disjoint B has 10 through epoch30. States, not futures, are independent sampling units.

Positive Delta_h = L_Wait − L_Now favors Now. rho(S) is the future-conditional mean of (Delta1+Delta2+Delta3)/3. tau(S) is the mean of per-future relative differences using branch losses averaged over epochs28–30. Phase 5B estimates across-state covariance psi, not correlation or predictive accuracy. The frozen nested bootstrap resamples states and complete future records; intervals spanning zero are unresolved.

[Full Phase 5 specification](../PHASE5_DESIGN.md) contains frozen estimands and seed rules. [Final snapshot](FINAL_SCIENTIFIC_SNAPSHOT.md) explains follow-up LR switching, inherited-momentum Reset, composite m=4 averaging, A1-only exact replay, MNIST screening and Wait-d. These follow-ups do not replace primary outcomes. MNIST and Wait-d executed definitions are retained in their source modules and saved summaries; no standalone protocol was found.

The public copy removes private paths, execution infrastructure and raw checkpoints. Historical hash/authorization checks are not portable execution grants; see [reproduction limits](../reproduce/README.md).
