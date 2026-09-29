# Local public-release candidate — 2026-09-29

## Public-release status

**NOT READY for GitHub publication: license choice and public author metadata remain unconfirmed.** Documentation, source selection, frozen summaries, lightweight tests and local privacy checks have been prepared. No remote operation, upload, new experiment, scientific resampling or inferential recomputation was performed.

## Preservation and Git

The internal main branch has freeze commit `97bef5f` (`docs: freeze final ReflexML scientific state`). Exactly seven intended freeze files were committed. Pre-existing branching.py, training.py and research-skill modifications and unrelated untracked material were preserved. The new global novelty note clearly records conversation-originated final authority, not a fabricated detailed audit.

The public-release branch is an orphan branch in a separate managed worktree. It does not inherit internal Git history. This avoids publishing data, private paths or old orchestration from reachable ancestors. Original history and scientific archive remain in the internal repository; the worktree still shares the local object store. A future publication should export this branch alone, never internal branches/tags or the shared object store. No remote was created or changed.

## Files added and modified

All public files are additions relative to the orphan branch; there is no parent tree to modify. Relative to internal sources, README is rewritten; Markdown paths/navigation and JSON provenance paths are privacy-adapted; selected source defaults replace personal locations with relative excluded-artifact placeholders. Scientific formulas, estimands, seeds, numerical results and intervals are preserved. Nine private-provenance-dependent tests receive explicit skip annotations; their internal originals are untouched.

The inventory is `results/SOURCE_MANIFEST.json`: 67 retained source-bound files with original and public hashes. New authoring consists of README, .gitignore, .gitattributes, docs/EXPERIMENT_DESIGN.md, this report, reproduce/README.md, results/README.md, scripts/validate_release.py, scripts/make_figures.py and three figures. There is no LICENSE or CITATION.cff pending user confirmation. No DOI, coauthor, affiliation or publication venue is invented.

Clean public document names contain the final snapshot, postmortem and global stopping provenance. Supporting frozen protocols remain at their existing root paths to avoid changing imports and discoverability. Earlier audit and protocol continuation language is historical and superseded by the final freeze. The public presentation copies do not replace internal hash-bound originals or certify exact execution portability.

## Intentionally excluded

- **Large artifacts:** downloaded datasets, full-state model/optimizer checkpoints, raw runs/order tapes, bootstrap arrays and ZIP archives. Seven internal working-tree files exceed 10 MB; one exceeds 50 MB (a screening ZIP). One large downloaded data file is tracked internally. None is in the public candidate. Additional protected artifacts outside the repository are intentionally not packaged.
- **Internal notes:** Codex skills/context packs, prompts, brainstorming, handoffs, control-plane and backup-provider machinery (including phase5_evidence.py, which requires private r5 source at import), authorization/candidate records, ledger logs and conversation caches.
- **Privacy/security:** personal absolute paths, machine mount locations, execution environment identities and local provenance paths. Public placeholders refer to excluded artifacts; they are not usable private locations.
- **Obsolete/intermediate outputs:** early Phase 2–4 generated results, superseded preflight trees, duplicate screening directories, unselected exploratory output and production launchers. Internal originals are preserved. Phase 5 execution/preflight test modules and a launcher-dependent LR-dynamics test module are not included because they depend on excluded execution infrastructure.

## Large-file decisions

All listed files remain internal and are excluded from public-release; no LFS is introduced. Sizes below use decimal MB.

| Internal relative path | MB | Tracked internally | Public decision |
| --- | ---: | --- | --- |
| data/MNIST/raw/train-images-idx3-ubyte | 47.04 | No | Exclude downloaded data |
| data/FashionMNIST/raw/train-images-idx3-ubyte | 47.04 | Yes | Exclude data and inherited history |
| data/FashionMNIST/raw/train-images-idx3-ubyte.gz | 26.42 | No | Exclude download archive |
| outputs/collapse_external_v0_1_n6.zip | 79.48 | No | Exclude raw archive |
| outputs/collapse_external_v0_1_n12/records.json | 32.66 | No | Exclude raw dump; retain saved analysis |
| outputs/collapse_external_v0_1_n6/records.json | 16.33 | No | Exclude raw dump; retain saved analysis |
| outputs/collapse_external_v0_1_n6 2/records.json | 16.33 | No | Exclude duplicate directory |

## Scientific consistency

All eight publication checks **PASS**: Phase 5A, Phase 5B, LR-switch, Momentum Reset, m=4, MNIST, Wait-d and final novelty. These check transcription and scope, not new scientific inference. JSON numeric leaves and selected CSV bytes were checked against original saved sources. N=36 versus N=12 and A-all-futures versus historical A1 remain distinct. Keep intervals use their own frozen bootstrap. Wait-d d4 movement count is 10/12; N remains 12. MNIST N=12 includes the original six states. Phase 5B remains unresolved; no zero association or unpredictability claim. MNIST remains AMBIGUOUS. m=4 keeps the disclosed endpoint preview before bootstrap-matrix materialization; full procedural compliance is not claimed.

Three figures show saved Phase 5 horizon estimates/intervals, historical A1 fixed-probe closure, and Wait-d pre/post means. Their titles, axes, sign conventions, scopes and captions are explicit. No interpolation is presented as a new measured trajectory or inferential timing law.

## Privacy/security audit

The internal scan inspected 1,753 text files up to 2 MB each, using credential, assignment, email, local-path, private-URL and phone patterns. No credential/secret assignment, email or private-URL patterns were found. Local-path patterns occurred 1,075 times; those files were excluded or adapted. Phone patterns occurred within numbers and hashes and were not used as evidence of personal phone numbers. No secret values are reported.

The public text set has no personal absolute paths, credential-pattern hits, email addresses or private URLs. Broad phone-pattern hits are numerical/hash substrings rather than contact metadata. Markdown local links resolve. No tracked candidate file exceeds 10 MB. This is a bounded pattern/content review, not proof that arbitrary obfuscated secrets cannot exist. Internal historical Git objects and large raw files were not exhaustively secret-scanned; they are unreachable from this orphan candidate and are not publication content.

## Validation

- Supported lightweight suite: 65 synthetic tests passed.
- Public retained-suite discovery (`python3 -m unittest discover -s tests`): 117 tests, 107 passed, 10 explicitly skipped, no failures. Five skips require excluded Momentum Reset historical provenance, four require excluded m=4 authority/data/order records; one existing skip requires the excluded real historical LR replay artifacts. An initial exploratory discovery exposed those missing dependencies and the omitted production launcher and historical m=4 authority setup; the public test selection/annotations now disclose them.
- Both retained CLIs passed --help without loading data or training.
- Python compile/import checks passed for retained modules, scripts and tests.
- Public hash, frozen transcription, sample-count, local-link and path checks passed.
- Figures were rendered from saved numeric outputs and visually inspected. Frozen CSV CRLF bytes and existing Markdown hard breaks are retained; ordinary Git whitespace warnings on them do not indicate numerical changes.

Local validation used Python 3.14, torch 2.12.0, torchvision 0.27.0, NumPy 2.4.4, SciPy 1.17.1 and Matplotlib 3.10.9. Existing requirements are lower bounds, not a frozen runtime lock. No dependency installation or new training was necessary. Full scientific reproduction remains unavailable without excluded artifacts and the original environment.

## Remaining publication blockers

1. Choose a license. MIT is recommended for a small research-code portfolio; Apache-2.0 is another standard option. No legal license was selected without confirmation.
2. Confirm the preferred public author name before adding CITATION.cff. No name is guessed.
3. User review before publication; this task deliberately performs no push. Missing raw artifacts are a disclosed reproduction limitation, not a claim of full archival completeness.

Recommended GitHub description (under 160 characters):

> Controlled learning-rate intervention study with paired stochastic futures, exact replay, bounded mechanism tests and explicit novelty-based stopping.

Recommended topics: `machine-learning`, `pytorch`, `learning-rate`, `sgd`, `reproducible-research`, `controlled-experiments`, `negative-results`, `research-portfolio`.

Recommended local commit message: `docs: prepare ReflexML public research release`.
