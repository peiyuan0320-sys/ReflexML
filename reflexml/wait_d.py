"""ReflexML Wait-d v0.1. Dedicated full reruns; no prefix reuse or retries.

Production entry points bind the frozen Phase 5 ledger, not caller supplied seeds.
Synthetic kernels below exist for deterministic implementation tests only.
"""
from copy import deepcopy
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from .branching import BranchingConfig, order_hashes, run_branch, states_equal
from .collapse_dynamics_replay import BASE_OUTPUT, LEDGER
from .data import make_loaders_from_datasets, make_split_indices
from .checkpoint import restore_checkpoint
from .model import FashionMLP
from .reproducibility import capture_rng_states, restore_rng_states
from .mechanism_lr_switch import (
    FROZEN_BRANCH_LEDGER_SHA256, _configuration, _evidence_map, load_unit,
    validate_production_dataset,
)
from .phase5 import sha256_file
from .stability import with_future_shuffle

PROTOCOL = 'ReflexML Wait-d v0.1'
STATES = tuple(range(1, 13))
FUTURES = (1, 2)
ARMS = {'Now': (0, 5), 'Wait1': (1, 2), 'Wait2': (2, 3),
        'Wait3': (3, 4), 'Wait4': (4, 5)}
MAPPING = 'phase5_preflight_r4/future_seed_mapping.csv'
BASE_MAPPING_SHA = '7ccadfe4d1291c2a34c8b1b3b52fc80ebd2f6f9eb9c05b360037043661dd51bf'
MAPPING_SHA = '4a9ba1b441c1825ec7a9d8c211b02164a5f90b3b4dbccf24749c035982660f21'
CODE_FILES = ('reflexml/wait_d.py', 'run_wait_d.py', 'reflexml/branching.py',
              'reflexml/checkpoint.py', 'reflexml/config.py', 'reflexml/data.py',
              'reflexml/model.py', 'reflexml/training.py', 'reflexml/stability.py',
              'reflexml/reproducibility.py', 'reflexml/phase5.py',
              'reflexml/mechanism_lr_switch.py', 'reflexml/collapse_dynamics_replay.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    # Never overwrite an existing attempt, validation, or analysis.
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def runtime_identity():
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'torch': str(torch.__version__), 'platform': platform.platform(),
            'device': 'cpu', 'num_workers': 0, 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
            'deterministic': torch.are_deterministic_algorithms_enabled()}


def schedules():
    return {arm: [0.1] * delay + [0.05] * (horizon - delay)
            for arm, (delay, horizon) in ARMS.items()}


def load_sources(repo):
    """Read only 24 hash-bound artifacts, after streaming frozen lineage once."""
    repo = Path(repo)
    if sha256_file(repo / MAPPING) != MAPPING_SHA:
        raise ValueError('Frozen future mapping hash mismatch')
    if sha256_file(LEDGER) != FROZEN_BRANCH_LEDGER_SHA256:
        raise ValueError('Frozen branch ledger hash mismatch')
    selected = {(s, r): [] for s in STATES for r in FUTURES}
    with LEDGER.open() as handle:
        for line in handle:
            event = json.loads(line)
            identity = event.get('scientific_identity', {})
            key = identity.get('base_run_id'), identity.get('replica_id')
            if identity.get('block') == 'A' and key in selected:
                selected[key].append(event)
    with (repo / MAPPING).open() as handle:
        mapping = {(int(r['base_run_id']), int(r['replica_id'])): r
                   for r in csv.DictReader(handle) if r['block'] == 'A'}
    if sha256_file(repo / 'phase5_preflight_r4/base_seed_manifest.csv') != BASE_MAPPING_SHA:
        raise ValueError('Frozen base seed mapping hash mismatch')
    with (repo / 'phase5_preflight_r4/base_seed_manifest.csv').open() as handle:
        bases = {int(r['base_run_id']): int(r['base_seed']) for r in csv.DictReader(handle)}
    registration = json.loads((repo / 'phase5_preflight_r4/dataset_identity.json').read_text())
    units, bindings = {}, {}
    for key, events in selected.items():
        s, r = key
        if [e['event_type'] for e in events] != [
                'attempt_started', 'artifact_produced', 'terminal_success']:
            raise ValueError('Incomplete or ambiguous historical lineage')
        first, produced, terminal = events
        if (len({e['attempt_id'] for e in events}) != 1 or
                any(e['scientific_identity'] != first['scientific_identity'] for e in events) or
                produced['artifact_sha256'] != terminal['artifact_sha256']):
            raise ValueError('Historical lineage mismatch')
        path = BASE_OUTPUT / f'base_{s:03d}_attempt_001/primary_state.json'
        primary = json.loads(path.read_text())
        checkpoint_path = BASE_OUTPUT / f'base_{s:03d}_attempt_001/epoch_014.pt'
        identity = first['scientific_identity']
        if (primary['base_run_id'] != s or primary['epoch'] != 14 or
                primary['base_seed'] != bases[s] or
                primary['dataset_identity_sha256'] != digest(registration) or
                primary['checkpoint_path'] != str(checkpoint_path) or
                primary['checkpoint_sha256'] != identity['checkpoint_sha256'] or
                primary['protocol_version'] != 'Phase5-v1' or
                primary['state_manifested_before_future_outcomes'] is not True):
            raise ValueError('Primary-state binding mismatch')
        future = mapping[key]
        if int(future['counter']) != 0:
            raise ValueError('Only frozen counter zero is allowed')
        unit = load_unit(repo, base_run_id=s, replica_id=r,
                        checkpoint_path=checkpoint_path,
                        checkpoint_sha256=primary['checkpoint_sha256'],
                        artifact_path=Path(produced['artifact_path']),
                        artifact_sha256=produced['artifact_sha256'])
        config = _configuration(unit.checkpoint)
        if (config.seed != bases[s] or unit.checkpoint['random_seed'] != bases[s] or
                unit.checkpoint['split_fingerprint'] != primary['split_fingerprint']):
            raise ValueError('Checkpoint seed/split mismatch')
        # Source bytes bind complete model, optimizer, RNG and generator state.
        for field in ('model_state_dict', 'optimizer_state_dict',
                      'train_loader_generator_state', 'python_rng_state',
                      'numpy_rng_state', 'torch_rng_state'):
            if field not in unit.checkpoint:
                raise ValueError(f'Incomplete checkpoint: {field}')
        units[key] = unit
        bindings[f'{s:03d}-A{r}'] = {
            'state': s, 'future': r, 'base_seed': bases[s],
            'source_path': str(checkpoint_path), 'source_sha256': unit.checkpoint_sha256,
            'primary_path': str(path), 'primary_sha256': sha256_file(path),
            'future_seed': unit.future_seed, 'namespace': future['namespace'], 'counter': 0,
            'scientific_identity': identity,
            'historical_path': produced['artifact_path'],
            'historical_sha256': unit.source_sha256, 'config': unit.checkpoint['config'],
            'split': primary['split_fingerprint'],
            'dataset_identity': primary['dataset_identity_sha256'], 'population_size': 60000}
    return units, bindings


def validate_source_state(checkpoint, data):
    """Restore complete source without updates; preserve caller RNG state."""
    from .branching import training_state
    rng = capture_rng_states()
    try:
        config = _configuration(checkpoint)
        model = FashionMLP(config.hidden_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        restore_checkpoint(deepcopy(checkpoint), model, optimizer, data.train_generator,
                           data.split_fingerprint)
        actual = training_state(model, optimizer, data.train_generator)
        expected = {'model': checkpoint['model_state_dict'],
                    'optimizer': checkpoint['optimizer_state_dict'],
                    'generator': checkpoint['train_loader_generator_state'],
                    'rng': {key: checkpoint[key] for key in rng}}
        if not states_equal(actual, expected):
            raise ValueError('Complete source restore mismatch')
        groups = optimizer.state_dict()['param_groups']
        state = optimizer.state_dict()['state']
        if len(groups) != 1 or groups[0]['lr'] != .1 or groups[0]['momentum'] != .9:
            raise ValueError('Source SGD metadata mismatch')
        parameters = list(model.parameters())
        ids = groups[0]['params']
        if len(ids) != len(parameters) or set(state) != set(ids):
            raise ValueError('Source momentum inventory mismatch')
        for identifier, parameter in zip(ids, parameters, strict=True):
            buffer = state[identifier].get('momentum_buffer')
            if (buffer is None or buffer.shape != parameter.shape or
                    buffer.dtype != parameter.dtype or not torch.isfinite(buffer).all() or
                    not torch.isfinite(parameter).all()):
                raise ValueError('Source tensor shape/dtype/finiteness mismatch')
    finally:
        restore_rng_states(rng)


def preflight(repo, dataset):
    repo = Path(repo)
    units, bindings = load_sources(repo)
    validate_production_dataset(repo, dataset)
    for unit in units.values():
        data = make_loaders_from_datasets(dataset, [], _configuration(unit.checkpoint))
        if data.split_fingerprint != unit.checkpoint['split_fingerprint']:
            raise ValueError('Dataset split mismatch')
        validate_source_state(unit.checkpoint, data)
    historical_runtime = json.loads(
        (repo / 'phase5_preflight_r3/preflight_manifest.json').read_text())['environment']
    runtime = runtime_identity()
    if any(runtime.get(k) != v for k, v in historical_runtime.items()):
        raise ValueError('Historical runtime compatibility mismatch')
    manifest = {'protocol': PROTOCOL, 'states': list(STATES), 'futures': list(FUTURES),
                'schedules': schedules(), 'trajectories': 120, 'arm_epochs': 456,
                'base_mapping_sha256': BASE_MAPPING_SHA, 'mapping_sha256': MAPPING_SHA, 'ledger_sha256': FROZEN_BRANCH_LEDGER_SHA256,
                'bindings': bindings, 'runtime': runtime,
                'code': {p: sha256_file(repo / p) for p in CODE_FILES},
                'historical_runtime_reference': {
                    'path': 'phase5_preflight_r3/preflight_manifest.json',
                    'sha256': sha256_file(repo / 'phase5_preflight_r3/preflight_manifest.json')},
                'dataset_registration_sha256': sha256_file(
                    repo / 'phase5_preflight_r4/dataset_identity.json')}
    return units, manifest


def run_unit(checkpoint, dataset, seed, *, on_arm=None):
    """Synthetic-testable kernel: each arm independently restores epoch14."""
    original = deepcopy(checkpoint)
    seeded = with_future_shuffle(checkpoint, seed)
    results = {}
    for arm, (delay, horizon) in ARMS.items():
        try:
            result = run_branch(seeded, dataset, BranchingConfig(horizon=horizon), 0.5,
                                delay_epochs=delay)
            if not states_equal(result['before'], results.get('Now', result)['before']):
                raise ValueError('Initial training state mismatch')
            if (result['orders'] != results.get('Now', result)['orders'][:horizon] or
                    [row['learning_rate'] for row in result['history']] != schedules()[arm]):
                raise ValueError('Order or LR schedule mismatch')
        except Exception as error:
            if on_arm is not None:
                on_arm(arm, None, error)
            raise
        results[arm] = result
        if on_arm is not None:
            on_arm(arm, result, None)
    if not states_equal(checkpoint, original) or not states_equal(
            seeded, with_future_shuffle(original, seed)):
        raise ValueError('Source checkpoint mutated')
    now = results['Now']['orders']
    for arm, result in results.items():
        if result['orders'] != now[:ARMS[arm][1]]:
            raise ValueError('Shared future order mismatch')
    return results


def run(repo, dataset, output):
    units, manifest = preflight(repo, dataset)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', manifest)
    for (s, r), unit in units.items():
        try:
            if preflight_identity(repo, manifest) is False:
                raise ValueError('Code/runtime changed before unit')
            source = Path(manifest['bindings'][f'{s:03d}-A{r}']['source_path'])
            if sha256_file(source) != unit.checkpoint_sha256:
                raise ValueError('Source bytes changed')
            def persist_arm(arm, result, error):
                binding = manifest['bindings'][f'{s:03d}-A{r}']
                if error is not None:
                    write_json(output / f'{s:03d}-A{r}-{arm}.json',
                               {'state': s, 'future': r, 'arm': arm, 'status': 'FAILURE',
                                'binding': binding, 'error': f'{type(error).__name__}: {error}'})
                    return
                record = {'state': s, 'future': r, 'arm': arm, 'status': 'SUCCESS',
                          'binding': binding,
                          'history': result['history'], 'orders': result['orders'],
                          'order_sha256': order_hashes(result['orders']),
                          'evidence': result['state_finiteness_evidence']}
                write_json(output / f'{s:03d}-A{r}-{arm}.json', record)

            run_unit(unit.checkpoint, dataset, unit.future_seed, on_arm=persist_arm)
            if sha256_file(source) != unit.checkpoint_sha256 or not preflight_identity(repo, manifest):
                raise ValueError('Source/code/runtime changed during unit')
        except Exception as error:
            write_json(output / f'{s:03d}-A{r}-failure.json',
                       {'status': 'FAILURE', 'state': s, 'future': r,
                        'error': f'{type(error).__name__}: {error}'})
            raise
    return output


def preflight_identity(repo, manifest):
    return (manifest['runtime'] == runtime_identity() and
            manifest['code'] == {p: sha256_file(Path(repo) / p) for p in CODE_FILES})


def compare_reproduction(record, historical):
    """Exact historical h1..h4 comparison; no claim about absent RNG snapshots."""
    prefix = {'Now': 'now', 'Wait3': 'wait'}[record['arm']]
    history = historical[prefix + '_history']
    evidence = historical[prefix + '_state_finiteness_evidence']
    orders = historical[prefix + '_realized_orders']
    if len(history) != 4 or len(evidence) != 4 or len(orders) != 4:
        raise ValueError('Historical coverage missing')
    if record['history'][:4] != history or record['orders'][:4] != orders:
        raise ValueError('Historical history/order mismatch')
    if historical[prefix + '_epoch_order_sha256'] != order_hashes(orders):
        raise ValueError('Historical order hash mismatch')
    for new, old in zip(record['evidence'][:4], evidence, strict=True):
        if new['epoch'] != old['epoch']:
            raise ValueError('Historical evidence epoch mismatch')
        for group in ('model_parameters', 'optimizer_state'):
            left, right = _evidence_map(new[group]), _evidence_map(old[group])
            if left.keys() != right.keys() or not left:
                raise ValueError('Historical tensor inventory mismatch')
            for key in left:
                a, b = left[key], right[key]
                if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                    raise ValueError('Historical tensor dtype/shape/value mismatch')
    return True


def validate_records(records, bindings, historical):
    """Fail closed before computing any scientific quantity."""
    try:
        expected = {(s, r, a) for s in STATES for r in FUTURES for a in ARMS}
        keys = [(row['state'], row['future'], row['arm']) for row in records]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('Coverage/duplicate mismatch')
        indexed = dict(zip(keys, records, strict=True))
        for (s, r, arm), row in indexed.items():
            h = ARMS[arm][1]
            if row['status'] != 'SUCCESS' or row['binding'] != bindings[f'{s:03d}-A{r}']:
                raise ValueError('Source/future/dataset/config/split identity mismatch')
            if ([x['epoch'] for x in row['history']] != list(range(15, 15 + h)) or
                    [x['learning_rate'] for x in row['history']] != schedules()[arm] or
                    not all(np.isfinite(x['val_loss']) for x in row['history'])):
                raise ValueError('Loss/epoch/LR mismatch')
            if (len(row['orders']) != h or len(row['evidence']) != h or
                    row['order_sha256'] != order_hashes(row['orders']) or
                    row['orders'] != indexed[s, r, 'Now']['orders'][:h]):
                raise ValueError('Actual order/boundary/evidence mismatch')
            binding = row['binding']
            config = binding['config']
            indices, _ = make_split_indices(binding['population_size'], config['train_size'],
                                            config['val_size'], config['split_seed'])
            batch_size = config['batch_size']
            lengths = [min(batch_size, len(indices)-i) for i in range(0, len(indices), batch_size)]
            for order in row['orders']:
                if ([len(batch) for batch in order] != lengths or
                        sorted(sum(order, [])) != sorted(indices)):
                    raise ValueError('Minibatch population/boundary mismatch')
            reference = historical[s, r]['now_state_finiteness_evidence'][0]
            for epoch, evidence in enumerate(row['evidence'], 15):
                if evidence['epoch'] != epoch:
                    raise ValueError('Evidence epoch mismatch')
                for group in ('model_parameters', 'optimizer_state'):
                    values, structure = _evidence_map(evidence[group]), _evidence_map(reference[group])
                    if values.keys() != structure.keys():
                        raise ValueError('Tensor inventory mismatch')
                    for name, value in values.items():
                        if (value.dtype != structure[name].dtype or value.shape != structure[name].shape
                                or not torch.isfinite(value).all()):
                            raise ValueError('Tensor structure/nonfinite mismatch')
            if arm in ('Now', 'Wait3'):
                compare_reproduction(row, historical[s, r])
        return {'status': 'PASS', 'classification': None,
                'historical_now': 'PASS', 'historical_wait3': 'PASS'}
    except (KeyError, ValueError, TypeError, IndexError, RuntimeError) as error:
        return {'status': 'FAIL', 'classification': 'INVALID', 'error': str(error)}


def summarize(delta):
    """Frozen algebra for already validated data; shape state,future,delay,horizon.

    delta stores pre/post gaps in its last axis (12,2,4,2).
    """
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (12, 2, 4, 2) or not np.isfinite(delta).all():
        raise ValueError('Exact finite 12x2x4x2 delta required')
    rows = {}
    for j, d in enumerate(range(1, 5)):
        p_i, q_i = delta[:, :, j, :].mean(axis=1).T
        p, q = float(p_i.mean()), float(q_i.mean())
        se = float(p_i.std(ddof=1) / np.sqrt(12))
        split = delta[:, :, j, :].mean(axis=0)
        s = [bool(pr > 0 and qr < pr) for pr, qr in split]
        valid = bool(p > max(0.01, 2 * se) and np.all(split[:, 0] > 0))
        c = float(1 - q / p) if valid else None
        movement = (p_i > 0) & (np.abs(q_i) < np.abs(p_i))
        rows[d] = {'P_i': p_i.tolist(), 'Q_i': q_i.tolist(), 'P': p, 'Q': q,
                   'A': q - p, 'SE_state': se, 'denominator_threshold': max(.01, 2*se),
                   'denominator_checks': {'aggregate_above_threshold': p > max(.01, 2*se),
                                          'A1_positive': bool(split[0, 0] > 0),
                                          'A2_positive': bool(split[1, 0] > 0)},
                   'denominator_valid': valid, 'C': c,
                   'M_i': movement.astype(int).tolist(), 'n': int(movement.sum()),
                   'overshoot_i': ((p_i > 0) & (q_i < 0)).tolist(),
                   'split': {f'A{k+1}': {'P': float(pr), 'Q': float(qr), 'S': s[k]}
                             for k, (pr, qr) in enumerate(split)},
                   'split_conflict': bool(np.any(split[:, 0] <= 0) or s[0] != s[1])}
    t = [d for d, row in rows.items() if row['denominator_valid'] and
         row['C'] >= 0.25 and row['n'] >= 9]
    non3 = [d for d in t if d != 3]
    interpretable = sum(row['denominator_valid'] for row in rows.values())
    predicates = {
        'INVALID': False,
        'GO': interpretable >= 3 and len(t) >= 3 and len(non3) >= 2 and
              sum(all(rows[d]['split'][r]['S'] for r in ('A1', 'A2')) for d in non3) >= 2,
        'NO_GO_NO_NON_D3_CLOSURE': all(rows[d]['denominator_valid'] and
              rows[d]['C'] <= 0 and rows[d]['n'] == 0 and
              not rows[d]['split_conflict'] for d in (1, 2, 4)),
        'NO_GO_WAIT3_SPECIFIC': 3 in t and all(rows[d]['denominator_valid'] for d in (1, 2, 4))
              and len(non3) <= 1 and not any(row['split_conflict'] for row in rows.values())}
    checks = {
        'GO': {'three_interpretable': interpretable >= 3, 'three_substantial': len(t) >= 3,
               'two_non_d3_substantial': len(non3) >= 2,
               'two_non_d3_split_directions': sum(
                   all(rows[d]['split'][r]['S'] for r in ('A1', 'A2')) for d in non3) >= 2},
        'NO_GO_WAIT3_SPECIFIC': {'d3_substantial': 3 in t,
               'non_d3_denominators': all(rows[d]['denominator_valid'] for d in (1, 2, 4)),
               'at_most_one_non_d3_substantial': len(non3) <= 1,
               'no_split_conflict_all_delays': not any(row['split_conflict'] for row in rows.values())},
        'NO_GO_NO_NON_D3_CLOSURE': {
            d: {'denominator_valid': rows[d]['denominator_valid'], 'nonpositive_C':
                rows[d]['C'] is not None and rows[d]['C'] <= 0, 'zero_movement': rows[d]['n'] == 0,
                'no_split_conflict': not rows[d]['split_conflict']} for d in (1, 2, 4)}}
    label = next((k for k, value in predicates.items() if value), 'AMBIGUOUS')
    predicates['AMBIGUOUS'] = label == 'AMBIGUOUS'
    secondary = None
    if label == 'GO':
        cs = [row['C'] for row in rows.values() if row['denominator_valid']]
        span = max(cs) - min(cs)
        secondary = {'range_C': span, 'label': 'COARSE_DELAY_VARIATION' if span >= 0.25
                     else 'BROADLY_SIMILAR_CLOSURE'}
    return {'Delta_pre_post': delta.tolist(), 'delays': rows, 'T': t,
            'predicates': predicates, 'classifier_checks': checks, 'interpretable_count': interpretable,
            'classification': label, 'secondary': secondary}


def analyze_records(records, bindings, historical):
    gate = validate_records(records, bindings, historical)
    if gate['status'] != 'PASS':
        return {'classification': 'INVALID', 'validation': gate,
                'predicates': {'INVALID': True}, 'scientific_predicates_evaluated': False}
    indexed = {(x['state'], x['future'], x['arm']): x for x in records}
    delta = np.empty((12, 2, 4, 2))
    all_delta = {}
    for s in STATES:
        for r in FUTURES:
            now = indexed[s, r, 'Now']['history']
            for d in range(1, 5):
                wait = indexed[s, r, f'Wait{d}']['history']
                differences = [w['val_loss'] - n['val_loss'] for w, n in zip(wait, now)]
                all_delta[f'{s:03d}-A{r}-Wait{d}'] = differences
                delta[s-1, r-1, d-1] = differences[d-1:d+1]
    return {**summarize(delta), 'Delta': all_delta, 'validation': gate}


def inspect_run(repo, dataset, output, *, analyze=False):
    output = Path(output)
    try:
        units, manifest = preflight(repo, dataset)
        recorded = json.loads((output / 'manifest.json').read_text())
        if recorded != manifest:
            raise ValueError('Manifest/source/code/runtime identity changed')
        expected = {f'{s:03d}-A{r}-{a}.json' for s in STATES for r in FUTURES for a in ARMS}
        actual = {p.name for p in output.glob('*-A*.json')}
        if actual != expected:
            raise ValueError('Incomplete/extra/failure arm records')
        records = [json.loads((output / name).read_text()) for name in sorted(expected)]
        historical = {key: unit.historical for key, unit in units.items()}
        result = (analyze_records(records, manifest['bindings'], historical) if analyze else
                  validate_records(records, manifest['bindings'], historical))
        result['input_sha256'] = {name: sha256_file(output / name)
                                  for name in sorted(expected | {'manifest.json'})}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        result = {'status': 'FAIL', 'classification': 'INVALID', 'error': str(error)}
    write_json(output / ('analysis.json' if analyze else 'validation.json'), result)
    return result
