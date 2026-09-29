<!-- Public presentation copy: local paths/links adapted; internal original preserved. -->

# ReflexML Momentum-Reset v1 formal protocol freeze record

## Protocol identity

- Protocol ID: `ReflexML-Momentum-Reset-v1`
- Protocol path: `internal-artifacts/MOMENTUM_RESET_PROTOCOL.md`

## Authoritative pre-freeze candidate

- Status: `FREEZE CANDIDATE`
- Byte size: `20267`
- SHA-256: `e1c609a75f00219cf8fa03c3825d9bd7a6a692bccc55fb41373733a8fdfee95b`
- `execution_authorized = false`

The path, Protocol ID, status, byte size, SHA-256, and execution authorization were verified before editing.

## Scientific closure authority

The supplied independent scientific closure verdict was:

`MOMENTUM-RESET SCIENTIFIC CLOSURE PASS — READY FOR FREEZE`

- P0 = none
- P1 = none
- P2 = none

This verdict supports formal protocol freeze. It does not authorize experiment execution. This freeze record does not repeat the scientific audit.

## Frozen protocol authority

- Status: `FROZEN`
- Byte size: `20256`
- SHA-256: `5f6488f867c5a5663067ef5d5a85aea3aeca295e53260b42b9a21a61dac52f38`
- `execution_authorized = false`

## Mechanical-diff attestation

The direct candidate-to-frozen comparison contains exactly these five textual changes:

1. Title: `scientific protocol freeze candidate` → `frozen scientific protocol`.
2. Status field: `FREEZE CANDIDATE` → `FROZEN`.
3. Scope status sentence: `This document is a candidate only; its bytes and SHA-256 have not been frozen.` → `This document is frozen; its bytes and SHA-256 are recorded in MOMENTUM_RESET_PROTOCOL_FREEZE_RECORD.md.` (the filename is code-formatted in the protocol).
4. Section 13 heading: `candidate status` → `frozen status`.
5. Final status paragraph: `This document remains FREEZE CANDIDATE` → `This document is FROZEN`, and its closing reference `this candidate` → `this protocol`; `execution_authorized = false` and the prohibition on implementation, Reset training, and Reset-outcome inspection remain in place.

All other protocol bytes are unchanged. No scientific content changed during freeze. The Keep source authority, Reset treatment, `N = 36`, `K = 2`, A1/A2 identities, primary epoch18 validation-loss outcome, estimands, aggregation, bootstrap, interpretation, retry/non-finite rules, and compatibility requirement remain unchanged.

## Remaining execution blockers

Formal protocol freeze does not authorize experiment execution.

Reset execution remains blocked pending at least:

1. Implementation.
2. Independent implementation audit.
3. Numerical training-kernel/environment compatibility gate PASS against the existing Keep cells.
4. Required state, RNG, treatment, and provenance integrity gates.
5. Explicit later execution authorization.

The numerical training-kernel/environment compatibility execution gate has **not** yet passed. No code change, training, Reset branch, Keep rerun, or primary or secondary analysis was performed in this freeze task.

This record intentionally omits its own SHA-256 to avoid self-reference. Its byte size and SHA-256 are reported externally after creation.
