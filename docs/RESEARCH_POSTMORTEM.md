<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# ReflexML research postmortem

**COMPLETE / FROZEN — 2026-09-29 (Asia/Shanghai).** This is a retrospective judgment about research choices, not a new analysis. The [final scientific snapshot](FINAL_SCIENTIFIC_SNAPSHOT.md) is authoritative for findings and their limits. The project is retained as a controlled empirical study, undergraduate research project, and reproducibility/research-methods portfolio. No further scientific experiments are authorized under the current question.

## What went well

- **Protocol discipline.** Phase 5 separated its questions, froze estimands and sample allocation, used outcome-independent states, and prevented secondary analyses from replacing primary classifications.
- **Paired stochastic futures.** Same-state branches retained optimizer state and matched future minibatch orders. Disjoint A/B measurement blocks separated short response from final effect. Replicas were not counted as independent states.
- **Exact replay.** Identity/order/model/optimizer/endpoint checks made it possible to distinguish a faithful historical trajectory from a merely similar endpoint. Sparse replay also exposed the limits of epoch-level stories.
- **Explicit estimands and interpretation.** Wait−Now signs, per-replica final ratios, state-level covariance, and factorial contrasts gave the study inspectable targets. Endpoint LR causality was distinguished from explaining an entire trajectory.
- **Negative-result preservation.** Phase 5B precision remained insufficient, Momentum Reset did not become an equivalence claim, m=4 remained composite, and MNIST remained AMBIGUOUS. These boundaries are part of the result.
- **Willingness to stop.** The final contribution audit was allowed to end expansion. A completed controlled study has value even when it does not justify a novel paper claim.

## What went wrong

### 1. Problem selection was too phenomenon-driven

The project began more from “what interesting effect can we study in this controllable system?” than from “what important literature gap already exists?” A convenient intervention system made questions experimentally accessible, but accessibility did not establish importance. An interesting local loss curve was too easily treated as the starting point for a research mainline.

The lesson is to require an important unresolved question before building a large experimental program around the system. A controllable setup can then serve that question rather than generate an indefinite sequence of phenomena.

### 2. Novelty review was too narrow

Earlier novelty reasoning often emphasized “has someone done this exact phenomenon?” rather than “does the remaining difference constitute a meaningful contribution beyond the broad prior phenomenon?” Different datasets, exact horizons, paired implementation details, and curve shapes can avoid an exact match while remaining instances of a known transition/catch-up story.

A **collision test** asks whether prior work already performed an identical or structurally near-identical experiment. A **contribution test** asks whether the difference left after acknowledging the strongest prior work is important, informative, and adequately supported. Passing the collision test does not pass the contribution test. The later audit exposed this distinction; its final project-level judgment governs the stop decision.

### 3. Mainline drift

New results repeatedly generated new questions: unstable labels suggested future replication; strong short responses suggested association; collapse suggested mechanism and temporal localization; residual shape differences suggested further discriminators. Individual pivots were often locally reasonable, but there was no sufficiently strict rule for promoting a secondary observation into the new mainline.

Follow-ups therefore risked inheriting relevance from their ancestry rather than independently earning it. A secondary observation should retain that status until its importance, novelty, measurability and tractability pass fresh gates. This does not retroactively invalidate the frozen follow-ups; it limits what they collectively claim.

### 4. Local rationality vs global value

Many experiments answered legitimate narrow questions. LR switching established an immediate controlled effect; Reset and m=4 constrained explanations; Wait-d broadened the local boundary finding. Yet a sequence of defensible experiments did not necessarily raise the final publication-value ceiling.

The historical K_A precision recommendation illustrates the distinction. More replicas could improve a statistical measurement, while the final contribution/value review still justifies **DO NOT EXPAND K_A**. Local information gain is not sufficient justification for continuing the whole project.

### 5. Rigor was optimized before research value was fully validated

Substantial effort went into reproducibility, execution integrity, exact identities and auditing before novelty and measurability were sufficiently de-risked. The rigor was useful: it made the evidence credible and prevented inappropriate reinterpretation. The mistake was sequencing, not rigor itself.

Early engineering effort should match the uncertainty that matters at the current decision. Validate the question's value and measurement prospects with a bounded pilot before investing in major execution infrastructure. Once a valuable experiment is selected, preserve rigorous controls and provenance appropriate to its claims.

### 6. Phase 5B measurability risk

The state-level question was conceptually interesting, but between-state signal relative to within-state future-randomness noise was not sufficiently de-risked before the larger experiment. A strongly positive average rho did not guarantee that states could be ranked reliably. N=36 with K_A=15/K_B=10 left association precision insufficient; exploratory decomposition found unresolved positive rho heterogeneity and a tau reliability-style estimate near 0.27.

This is not evidence that state effects do not exist or that long-term response is inherently unpredictable. The lesson is to estimate both between-unit and within-unit variation before committing to an association or predictor mainline, and to preserve the possibility that the desired quantity cannot be measured cheaply enough.

## Closure lesson

ReflexML produced credible scoped findings and reusable research assets. Its final limitation is the contribution ceiling: broad prior art, unresolved state-level identification, bounded mechanism constraints, and ambiguous transfer made additional experiments insufficiently valuable. The principal stopping reason is that marginal scientific novelty and expected value no longer justified expansion; compute unavailability is not the explanation.

The protocol below applies to **future, separate self-directed projects**. It does not reopen ReflexML, authorize pilots here, or propose another ReflexML direction.

# Research Selection Protocol v1

Future self-directed projects must pass these gates before substantial experimentation. Record the answer and evidence for each gate; downgrade or stop when a gate fails rather than treating implementation progress as a substitute.

## Gate 1 — Importance

Why is the question worth knowing? Identify who would learn something consequential, what uncertainty matters, and how the answer would change understanding or a decision. Experimental convenience and an interesting curve are insufficient by themselves.

## Gate 2 — Literature landscape

Identify the strongest near-neighbor work, not just exact lexical matches. Compare question, intervention, unit, assumptions, stochastic futures, measurement resolution and claim coverage. Record both what is already established and what remains uncertain; distinguish a collision test from a contribution test.

## Gate 3 — Contribution sentence

Force one sentence of this form:

> Existing work establishes X. It does not establish Y. This project will test Y using Z.

If Y is merely a different dataset, exact curve shape, terminology change, or local numerical instance, reject or downgrade to a replication/learning project. Z must distinguish Y from the strongest existing explanation, and the proposed claim must be no broader than the test can support.

## Gate 4 — Measurability

Run a cheap, bounded pilot to estimate:

- Signal size.
- Between-unit variation.
- Within-unit stochastic variation.
- Approximate measurement reliability and achievable precision.

Use the intended independent unit. Keep repeated futures distinct from additional units, and separate a strong population mean from reliable between-unit ranking. State assumptions and uncertainty. If a meaningful target cannot be resolved at a tractable cost, stop or downgrade before substantial experimentation.

## Gate 5 — Minimal discriminator

Identify the cheapest experiment that could kill the hypothesis. Specify competing explanations, what outcomes would distinguish them, and what each outcome would leave unresolved. Prefer a test that reduces the plausible stories over one that merely adds another interesting measurement.

## Gate 6 — Mainline promotion

A new observation may become the mainline only if at least one is true:

- The original question is resolved.
- The original question is infeasible.
- The new question is clearly more important.
- The new question is required to explain the core result.

AND the new question independently passes:

- Importance.
- Novelty.
- Measurability.
- Tractability.

Record the promotion decision explicitly. A surprising result or a locally reasonable follow-up alone is insufficient. Preserve the original primary outcome and distinguish follow-up evidence from it.

## Gate 7 — Stop rule

Before running the major experiment, specify what result would cause **STOP / FREEZE / PIVOT**. Include scientific claim criteria, precision/identifiability failures, and contribution/value limits, with a bounded effort budget. Define what remains reportable under negative or inconclusive outcomes.

Do not allow sunk cost to alter the stop criterion. A pivot must pass the selection gates anew; a freeze preserves the findings and limits without inventing a publication claim.
