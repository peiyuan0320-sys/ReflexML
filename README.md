# ReflexML

Controlled experiments on short-horizon learning-rate interventions and post-reunification attenuation in neural-network training.

**Status: COMPLETE / FROZEN** — **2026-09-29**

A completed controlled empirical study and research portfolio, with paired stochastic futures, frozen protocols, exact replay, partial external replication and explicit novelty-based stopping.

## Research question

Two branches start from the same full training checkpoint. One lowers the learning rate immediately; the other delays the reduction. How large is the immediate effect, how does it evolve, what remains when schedules reunify, and does a state's short response contain reliable information about its later effect?

## Experimental design

Phase 5 uses 36 outcome-independent Fashion-MNIST states. Paired branches share future minibatch orders. Separate future blocks measure short responses (15 per state) and final effects (10 per state). Frozen estimands distinguish average effects from state-level association; repeated futures do not add independent states.

```text
                 same full checkpoint S
                         |
                 +-------+-------+
                 |               |
                Now           Wait-d
             LR .05 now    LR .10 for d epochs
                 |          then LR .05
                 +-------+-------+
               matched future minibatch order
                   compare validation loss
```

See the [design guide](docs/EXPERIMENT_DESIGN.md) and [frozen protocol](PHASE5_DESIGN.md).

## Main findings

Positive loss contrasts favor Now or LR .05. Findings are conditional on the tested settings.

| Study | Frozen finding | Limit |
| --- | --- | --- |
| Phase 5A | Strong h1–3 gap (mean 0.04847); h4 mean 0.0000903, near-complete attenuation | Near zero is not equivalence or history forgetting |
| LR-switch | Same-S17 endpoint LR contrast 0.04110; 95% CI [0.03385, 0.04871] | Endpoint causality does not explain the whole historical trajectory |
| Momentum Reset | LR benefit survives inherited-buffer Reset (0.04440) | Inherited momentum alone is insufficient; primary interaction unresolved |
| m=4 | Contrast falls from 0.04842 to 0.01173 and remains positive | Composite gradient/batch-exposure intervention, not pure noise causation |
| Wait-d | Reunification-linked attenuation across d=1–4; N=12, K=2; frozen GO | Screening label, not a timing law or continuation permission |
| MNIST | Partial structural attenuation; final N=12, K=2 classification **AMBIGUOUS** | Strong collapse did not replicate |
| Phase 5B | rho–tau covariance CI spans zero | Insufficient precision, not absence of association or proof of unpredictability |

[Exact frozen values and claim limits](docs/FINAL_SCIENTIFIC_SNAPSHOT.md) · [selected source summaries](results/README.md)

![Phase 5 horizon response](results/figures/phase5_horizons.png)

Validation-loss difference Wait3−Now, N=36, K_A=15; bars are saved pointwise 95% intervals. Positive values favor Now. h4 follows schedule reunification; no exact-zero claim.

## What the project does not claim

ReflexML does not establish a new general LR scheduling law, complete training-history forgetting, momentum irrelevance, pure gradient-noise causation, a timing law over Wait-d, reliable prediction from rho(S) to tau(S), or dataset identity as the cause of Fashion-MNIST/MNIST differences. It introduces no novel optimizer, theory of SGD or causal-inference method. Unresolved and negative findings remain part of the evidence.

## Why the project stopped

Empirical effects were reproducible and informative, but adversarial literature review found that the remaining scientific differences did not support a sufficiently novel paper-level contribution at acceptable marginal cost. The final classification is **ONLY AS A REPLICATION / CONTROLLED EMPIRICAL STUDY**. The project froze rather than expanding experiments to manufacture a stronger claim.

[Scientific snapshot](docs/FINAL_SCIENTIFIC_SNAPSHOT.md) · [novelty and stopping provenance](docs/NOVELTY_AND_STOPPING.md) · [research postmortem](docs/RESEARCH_POSTMORTEM.md)

## Reproduction

Lightweight validation and synthetic fixture tests are available; see [verified commands and limits](reproduce/README.md). Full scientific reproduction requires excluded checkpoints, raw paired records, provenance and the original environment. This repository is not a self-contained exact-replay archive. Figures only render frozen saved numbers.

## Repository map

- `reflexml/`: retained Python model, checkpoint, branching, estimand and follow-up implementations.
- `scripts/`: public validation and frozen-number figure rendering.
- `tests/`: selected synthetic unit and integrity tests.
- `results/summaries/`: selected saved JSON/CSV results; `results/figures/`: three explanatory figures.
- `docs/`: final synthesis, design, stopping decision, postmortem and selected evidence reports.
- `reproduce/`: supported commands and excluded-artifact disclosure.

Publication is pending license choice and public author confirmation. No DOI, venue or institutional endorsement is claimed.
