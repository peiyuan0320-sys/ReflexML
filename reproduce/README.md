# Reproduction and validation

## Lightweight checks

From the repository root, use Python 3.11 or newer:

```sh
python3 -m pip install -r requirements.txt
python3 scripts/validate_release.py
python3 -m unittest tests.test_reproducibility tests.test_branching tests.test_collapse_external tests.test_wait_d
python3 run_collapse_external.py --help
python3 run_wait_d.py --help
```

The unit tests use small synthetic tensors and mocked boundaries; they do not download data or run scientific experiments. The verifier checks file hashes, frozen result transcription, sample counts, Markdown links and privacy patterns without recomputing inference. The figure script renders already saved estimates and intervals:

```sh
python3 scripts/make_figures.py
```

## Full scientific reproduction

This edition is not a bit-for-bit self-contained execution archive. It includes source, frozen definitions, selected primary JSON/CSV summaries and public presentation copies of reports. Data downloads, full-state checkpoints/optimizer/RNG snapshots, paired order tapes, raw runs, execution ledgers, protected provenance and environment-bound authorization records are excluded. Exact replay and recomputation of all primary statistics require those internal artifacts and the frozen environment. Do not bypass a historical integrity gate using edited public files. Embedded source hashes refer to internal originals; public adaptations have separate hashes in results/SOURCE_MANIFEST.json.

Source modules expose branching, analysis, screening and replay logic. run_collapse_external.py and run_wait_d.py preserve historical CLIs for inspection; only --help is verified here. Production runners and obsolete orchestration are excluded. Full execution is unavailable from this release and is not authorized by the completed project.

Dependencies use the existing broad requirements, not a historical environment lock. See [validation evidence](../docs/PUBLIC_RELEASE_REPORT.md) for the tested local versions. Installing on another machine may resolve other versions; no exact runtime portability is claimed.

Production r5 helpers in phase5.py depend on excluded phase5_evidence.py and private execution infrastructure. Those paths are not supported by this edition. Import checks cover retained modules; synthetic tests exercise only the documented supported subset.
