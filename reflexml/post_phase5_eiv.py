"""Post-Phase5-EIV-v3.1 exploratory analysis and precision planning.

No training code is imported. Importing this module does not read frozen data or
run analysis. Formal execution is explicit through main().
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew

PROTOCOL = Path(__file__).resolve().parents[1] / 'POST_PHASE5_EIV_PLAN.md'
PROTOCOL_SHA256 = '91cac4d95bf8d4606ef2c790efc22724d4723db6be413ecf2c147beeaad7369b'
RESULTS_PATH = PROTOCOL.parent / 'RESULTS_PHASE5.md'
REPLICA_PATH = Path('internal-artifacts/phase5_replica_estimates.csv')
STATE_PATH = Path('internal-artifacts/phase5_state_estimates.csv')
REPLICA_SHA256 = '1868b85cf9ac9f8ac6c158a8af7fa057f92a46ad492bf6a30942b098cace34d1'
STATE_SHA256 = 'd119d62176a780fe683ea7b38c630a21a4f0121e350420e7372fd536b1f0a056'
OUTER_NAMESPACE = 'ReflexML|PostPhase5|EIV-v3|outer-bootstrap'
HIER_NAMESPACE = 'ReflexML|PostPhase5|EIV-v3|hierarchical-bootstrap'
PRECISION_NAMESPACE = 'ReflexML|PostPhase5|EIV-v3|precision'
OUTER_SEED = 12479654370683736347
HIER_SEED = 3757477645905739108
OPT_OPTIONS = {'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 20000}
NEAR_BEST = 1e-6
QUANTILES = [0.025, 0.5, 0.975]

SCHEMAS = {
    'post_phase5_eiv_decomposition.csv': 'parameter observed_variance mean_measurement_variance raw_deconvolved_variance raw_signal_fraction outer_q025 outer_q500 outer_q975',
    'post_phase5_eiv_support.csv': 'parameter profile_region_id grid_value profile_loglik delta2loglik in_reference_support_set fit_status range_expansion_index endpoint_status',
    'post_phase5_eiv_bootstrap.csv': 'bootstrap_id sigma_rho2 sigma_tau2 psi r_latent r_defined fit_status boundary_flag',
    'post_phase5_eiv_loo.csv': 'excluded_base_run_id mu_rho mu_tau sigma_rho2 sigma_tau2 psi r_latent r_defined fit_status',
    'post_phase5_precision.csv': 'scenario_id latent_family measurement_engine N K_A K_B attempted_count valid_count failed_count cell_status incremental_cost true_psi mean_psi_hat bias mc_sd rmse median_interval_width q90_interval_width coverage precision_gain precision_gain_per_1000 pareto_dominated failure_types',
}
SCHEMAS = {name: columns.split() for name, columns in SCHEMAS.items()}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_identity(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f'SHA-256 mismatch: {path}')


def seeded_rng(namespace: str, *parts: object) -> tuple[np.random.Generator, dict[str, Any]]:
    label = '|'.join((namespace, *(str(part) for part in parts)))
    digest = hashlib.sha256(label.encode('utf-8')).digest()
    seed = int.from_bytes(digest[:8], 'big', signed=False)
    return np.random.Generator(np.random.PCG64(seed)), {
        'namespace': namespace, 'sha256': digest.hex(), 'seed': seed,
        'numpy_version': np.__version__,
    }


@dataclass(frozen=True)
class Data:
    ids: np.ndarray
    a: np.ndarray  # state x 15
    b: np.ndarray  # state x 10
    y: np.ndarray  # state x 2
    d: np.ndarray  # state x 2, estimated state-mean measurement variance


def summarize_replicas(ids: np.ndarray, a: np.ndarray, b: np.ndarray) -> Data:
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0] or len(ids) != a.shape[0]:
        raise ValueError('replica shape mismatch')
    if a.shape[1] < 2 or b.shape[1] < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid replica values')
    y = np.column_stack((a.mean(axis=1), b.mean(axis=1)))
    d = np.column_stack((a.var(axis=1, ddof=1) / a.shape[1], b.var(axis=1, ddof=1) / b.shape[1]))
    return Data(np.asarray(ids), a, b, y, d)


def load_inputs(replica_path: Path = REPLICA_PATH, state_path: Path = STATE_PATH) -> Data:
    check_identity(PROTOCOL, PROTOCOL_SHA256)
    result_record = RESULTS_PATH.read_text()
    for name, expected in ((REPLICA_PATH.name, REPLICA_SHA256),
                           (STATE_PATH.name, STATE_SHA256)):
        if f'| `{name}` | `{expected}` |' not in result_record:
            raise ValueError(f'frozen Phase 5 result record lacks expected CSV identity: {name}')
    check_identity(replica_path, REPLICA_SHA256)
    check_identity(state_path, STATE_SHA256)
    with replica_path.open(newline='') as handle:
        replica_rows = list(csv.DictReader(handle))
    with state_path.open(newline='') as handle:
        state_rows = list(csv.DictReader(handle))
    states: dict[int, dict[str, str]] = {}
    for row in state_rows:
        sid = int(row['base_run_id'])
        if sid in states:
            raise ValueError(f'duplicate state {sid}')
        states[sid] = row
    if len(states) != 36:
        raise ValueError('expected exactly 36 unique states')
    records: dict[tuple[int, str, int], dict[str, str]] = {}
    for row in replica_rows:
        sid, block = int(row['base_run_id']), row['block']
        rid = int(row['replica_id']) if row['replica_id'] else 0
        key = sid, block, rid
        if sid not in states or block not in ('A', 'B') or rid < 1 or key in records:
            raise ValueError(f'invalid or duplicate replica identity {key}')
        records[key] = row
    ids = np.array(sorted(states), dtype=int)
    a = np.empty((36, 15)); b = np.empty((36, 10))
    expected = {(sid, block, rid) for sid in ids for block, count in (('A', 15), ('B', 10)) for rid in range(1, count + 1)}
    if set(records) != expected:
        raise ValueError('missing or extra replica identities')
    for i, sid in enumerate(ids):
        for rid in range(1, 16):
            a[i, rid - 1] = float(records[sid, 'A', rid]['R'])
        for rid in range(1, 11):
            b[i, rid - 1] = float(records[sid, 'B', rid]['G'])
    data = summarize_replicas(ids, a, b)
    # Check every state summary derived from A/B replica records, including Phase 5 horizon fields.
    for i, sid in enumerate(ids):
        row = states[sid]
        values = {'rho_hat': data.y[i, 0], 'tau_hat': data.y[i, 1],
                  'rho_within_A_sd': np.std(a[i], ddof=1), 'tau_within_B_sd': np.std(b[i], ddof=1),
                  'rho_within_A_se': math.sqrt(data.d[i, 0]), 'tau_within_B_se': math.sqrt(data.d[i, 1])}
        if int(row['A_replica_count']) != 15 or int(row['B_replica_count']) != 10:
            raise ValueError(f'state count mismatch {sid}')
        for key, field in [('A34', 'A34'), ('deltaL1', 'deltaL1'), ('deltaL2', 'deltaL2'),
                           ('deltaL3', 'deltaL3'), ('deltaL4', 'deltaL4')]:
            samples = np.array([float(records[sid, 'A', rid][field]) for rid in range(1, 16)])
            values[('A34_mean_A' if key == 'A34' else key + '_mean_A')] = samples.mean()
            values[('A34_within_A_sd' if key == 'A34' else key + '_within_A_sd')] = samples.std(ddof=1)
        for name, value in values.items():
            if not np.isfinite(value) or not np.isclose(value, float(row[name]), rtol=1e-10, atol=1e-12):
                raise ValueError(f'state summary mismatch {sid}: {name}')
    return data


def decomposition(y: np.ndarray, d: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for col, label in enumerate(('rho', 'tau')):
        observed = float(np.var(y[:, col], ddof=1))
        measurement = float(np.mean(d[:, col]))
        raw = observed - measurement
        rows.append(dict(parameter=label, observed_variance=observed,
                         mean_measurement_variance=measurement, raw_deconvolved_variance=raw,
                         raw_signal_fraction=raw / observed if observed != 0 else 'NA'))
    return rows


def outer_only_bootstrap(data: Data, count: int = 20_000) -> tuple[list[dict], dict]:
    rng, meta = seeded_rng(OUTER_NAMESPACE)
    if meta['seed'] != OUTER_SEED:
        raise ValueError('outer seed mismatch')
    draws = np.full((count, 2), np.nan)
    failures = []; draw_outcomes = []
    for j in range(count):
        idx = rng.integers(0, len(data.ids), len(data.ids))
        try:
            values = np.array([r['raw_deconvolved_variance']
                               for r in decomposition(data.y[idx], data.d[idx])], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'non-finite raw variance components: {values.tolist()}')
            draws[j] = values
            draw_outcomes.append({'bootstrap_id': j + 1, 'status': 'valid',
                                  'raw_rho_variance': float(values[0]),
                                  'raw_tau_variance': float(values[1])})
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, FloatingPointError) as exc:
            failure = {'bootstrap_id': j + 1, 'failure': str(exc)}
            failures.append(failure)
            draw_outcomes.append({**failure, 'status': 'failed'})
    rows = decomposition(data.y, data.d)
    for col, row in enumerate(rows):
        valid = draws[np.isfinite(draws).all(axis=1), col]
        row.update(zip(('outer_q025', 'outer_q500', 'outer_q975'),
                       np.quantile(valid, QUANTILES, method='linear') if len(valid) else ('NA',) * 3))
    return rows, {'rng': meta, 'attempted_count': count, 'valid_count': count-len(failures),
                  'failed_count': len(failures), 'failures': failures,
                  'draw_outcomes': draw_outcomes,
                  'interpretation': 'outer-only percentile sensitivity range'}


def covariance_matrix(kind: str, theta: np.ndarray, value: float | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Exact PSD covariance and its derivatives; theta begins with both means."""
    z = theta[2:]
    if kind == 'unrestricted':
        a, b, c = z
        cov = np.array([[a*a, a*b], [a*b, b*b+c*c]])
        derivatives = [np.array([[2*a, b], [b, 0]]), np.array([[0, a], [a, 2*b]]),
                       np.array([[0, 0], [0, 2*c]])]
    elif kind == 'rho_fixed':
        if value == 0:
            c, = z; cov = np.diag([0., c*c]); derivatives = [np.diag([0., 2*c])]
        else:
            a = math.sqrt(value); b, c = z
            cov = np.array([[value, a*b], [a*b, b*b+c*c]])
            derivatives = [np.array([[0, a], [a, 2*b]]), np.diag([0., 2*c])]
    elif kind == 'tau_fixed':
        if value == 0:
            a, = z; cov = np.diag([a*a, 0.]); derivatives = [np.diag([2*a, 0.])]
        else:
            a, b = z; t = math.sqrt(value)
            cov = np.array([[a*a+b*b, b*t], [b*t, value]])
            derivatives = [np.diag([2*a, 0.]), np.array([[2*b, t], [t, 0]])]
    elif kind == 'both_zero':
        cov = np.zeros((2, 2)); derivatives = []
    elif kind == 'psi_fixed':
        if value == 0:
            a, c = z; cov = np.diag([a*a, c*c]); derivatives = [np.diag([2*a, 0.]), np.diag([0., 2*c])]
        else:
            u, q = z; a = math.exp(u); p = value
            cov = np.array([[a*a, p], [p, p*p/(a*a)+q*q]])
            derivatives = [np.diag([2*a*a, -2*p*p/(a*a)]), np.diag([0., 2*q])]
    elif kind == 'r_fixed':
        u, v = z; a, c = math.exp(u), math.exp(v); p = value*a*c
        cov = np.array([[a*a, p], [p, c*c]])
        derivatives = [np.array([[2*a*a, p], [p, 0.]]), np.array([[0., p], [p, 2*c*c]])]
    else:
        raise ValueError(f'unknown model {kind}')
    return cov, derivatives


def scientific_parameters(cov: np.ndarray, means: np.ndarray) -> dict[str, Any]:
    sr, st, psi = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
    defined = sr > 0 and st > 0
    return {'mu_rho': float(means[0]), 'mu_tau': float(means[1]),
            'sigma_rho2': sr, 'sigma_tau2': st, 'psi': psi,
            'r_latent': psi / math.sqrt(sr*st) if defined else None,
            'r_defined': defined, 'boundary_flag': sr == 0 or st == 0}


def loglik_and_gradient(theta: np.ndarray, y: np.ndarray, d: np.ndarray,
                        kind: str, value: float | None = None) -> tuple[float, np.ndarray, bool]:
    """Direct bivariate normal likelihood and analytic gradient of negative loglik."""
    try:
        cov, derivatives = covariance_matrix(kind, theta, value)
        if kind == 'r_fixed' and (cov[0, 0] <= 0 or cov[1, 1] <= 0):
            raise ValueError('correlation requires two positive latent variances')
        v = np.repeat(cov[None, :, :], len(y), axis=0)
        v[:, 0, 0] += d[:, 0]; v[:, 1, 1] += d[:, 1]
        np.linalg.cholesky(v)
        sign, determinant = np.linalg.slogdet(v)
        if not np.all(sign > 0) or not np.isfinite(determinant).all():
            raise ValueError('V_i not positive definite')
        residual = y - theta[:2]
        inv_residual = np.linalg.solve(v, residual[..., None])[..., 0]
        inv = np.linalg.inv(v)
        loglik = -0.5 * np.sum(2*math.log(2*math.pi) + determinant + np.einsum('ni,ni->n', residual, inv_residual))
        dcov = 0.5 * np.sum(inv - inv_residual[:, :, None]*inv_residual[:, None, :], axis=0)
        gradient = np.concatenate((
            -np.sum(inv_residual, axis=0),
            np.array([np.sum(dcov * partial) for partial in derivatives], dtype=float),
        ))
        if not np.isfinite(loglik) or not np.isfinite(gradient).all():
            raise ValueError('non-finite likelihood or gradient')
        return float(loglik), gradient, True
    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
        return float('-inf'), np.full(len(theta), np.nan), False


def starts_and_bounds(kind: str, y: np.ndarray, value: float | None = None) -> tuple[list[np.ndarray], list[tuple[float | None, float | None]]]:
    if kind in ('rho_fixed', 'tau_fixed') and (value is None or value < 0):
        raise ValueError('fixed variance must be nonnegative')
    if kind == 'r_fixed' and (value is None or not -1 <= value <= 1):
        raise ValueError('fixed correlation outside [-1, 1]')
    means = np.mean(y, axis=0).tolist()
    sr, st = np.std(y, axis=0, ddof=1)
    multipliers = (0.1, 0.5, 1.0); correlations = (-0.8, 0., 0.8)
    if kind == 'unrestricted':
        starts = [means + [qr*sr, r*qt*st, qt*st*math.sqrt(1-r*r)]
                  for qr, qt, r in product(multipliers, multipliers, correlations)]
        bounds = [(None, None), (None, None), (0, None), (None, None), (0, None)]
    elif kind == 'rho_fixed':
        if value == 0:
            starts = [means + [q*st] for q in (0., *multipliers)]
            bounds = [(None, None)]*2 + [(0, None)]
        else:
            starts = [means + [r*q*st, q*st*math.sqrt(1-r*r)] for q, r in product(multipliers, correlations)]
            bounds = [(None, None)]*3 + [(0, None)]
    elif kind == 'tau_fixed':
        if value == 0:
            starts = [means + [q*sr] for q in (0., *multipliers)]
            bounds = [(None, None)]*2 + [(0, None)]
        else:
            starts = [means + [q*sr*math.sqrt(1-r*r), r*q*sr] for q, r in product(multipliers, correlations)]
            bounds = [(None, None)]*2 + [(0, None), (None, None)]
    elif kind == 'both_zero':
        starts = [means]; bounds = [(None, None)]*2
    elif kind == 'psi_fixed':
        if value == 0:
            starts = [means + [qr*sr, qt*st] for qr, qt in product(multipliers, multipliers)]
            bounds = [(None, None)]*2 + [(0, None)]*2
        else:
            starts = []
            for qr, qt in product(multipliers, multipliers):
                dr, dt = qr*sr, qt*st
                sigma = max(dr, abs(value)/dt)
                q = math.sqrt(max(0., dt*dt-value*value/(sigma*sigma)))
                starts.append(means + [math.log(sigma), q])
            bounds = [(None, None)]*3 + [(0, None)]
    elif kind == 'r_fixed':
        starts = [means + [math.log(qr*sr), math.log(qt*st)] for qr, qt in product(multipliers, multipliers)]
        bounds = [(None, None)]*4
    else:
        raise ValueError(kind)
    return [np.array(start, dtype=float) for start in starts], bounds


def fit_model(y: np.ndarray, d: np.ndarray, kind: str = 'unrestricted', value: float | None = None,
              full_data: bool = False) -> dict[str, Any]:
    starts, bounds = starts_and_bounds(kind, y, value)
    attempts = []
    for start in starts:
        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            ll, grad, valid = loglik_and_gradient(theta, y, d, kind, value)
            return (-ll, grad) if valid else (float('inf'), np.zeros_like(theta))
        try:
            result = minimize(objective, start, method='L-BFGS-B', jac=True, bounds=bounds, options=OPT_OPTIONS)
            # Independent direct recomputation from the returned vector, never from result.fun.
            ll, _, pd = loglik_and_gradient(result.x, y, d, kind, value)
            cov, _ = covariance_matrix(kind, result.x, value)
            status = 'converged' if result.success and pd and math.isfinite(ll) else 'optimization_failure'
            science = scientific_parameters(cov, result.x[:2]) if pd else None
            if science is not None:
                science['boundary_flag'] = bool(science['boundary_flag'] or
                    (kind == 'r_fixed' and abs(value) == 1) or
                    (kind == 'unrestricted' and (result.x[2] == 0 or result.x[4] == 0)) or
                    (kind == 'rho_fixed' and value != 0 and result.x[-1] == 0) or
                    (kind == 'tau_fixed' and value != 0 and result.x[2] == 0) or
                    (kind == 'psi_fixed' and value != 0 and result.x[-1] == 0))
            attempts.append({'start': start.tolist(), 'returned_parameters': result.x.tolist(),
                             'optimizer_success': bool(result.success), 'optimizer_message': str(result.message),
                             'independent_loglik': ll if math.isfinite(ll) else None, 'positive_definite_V': pd,
                             'status': status, 'scientific_parameters': science})
        except (ValueError, OverflowError, FloatingPointError) as exc:
            attempts.append({'start': start.tolist(), 'returned_parameters': None, 'optimizer_success': False,
                             'optimizer_message': str(exc), 'independent_loglik': None, 'positive_definite_V': False,
                             'status': 'optimization_failure', 'scientific_parameters': None})
    good = [a for a in attempts if a['status'] == 'converged']
    if not good:
        return {'model': kind, 'fixed_value': value, 'fit_status': 'optimization_failure', 'starts': attempts,
                'best': None, 'near_best_ranges': None}
    best = max(good, key=lambda attempt: attempt['independent_loglik'])
    nearby = [a for a in good if best['independent_loglik'] - a['independent_loglik'] <= NEAR_BEST]
    names = ('mu_rho', 'mu_tau', 'sigma_rho2', 'sigma_tau2', 'psi', 'r_latent')
    ranges = {name: [min(vals), max(vals)] for name in names
              if (vals := [a['scientific_parameters'][name] for a in nearby if a['scientific_parameters'][name] is not None])}
    status = 'optimization_unstable' if full_data and kind == 'unrestricted' and len(nearby) < 3 else 'PASS'
    return {'model': kind, 'fixed_value': value, 'fit_status': status, 'starts': attempts,
            'best': best, 'near_best_ranges': ranges, 'near_best_count': len(nearby)}


def safe_fit(y: np.ndarray, d: np.ndarray, kind: str = 'unrestricted',
             value: float | None = None, full_data: bool = False) -> dict[str, Any]:
    try:
        return fit_model(y, d, kind, value, full_data)
    except (ValueError, OverflowError, FloatingPointError) as exc:
        return {'model': kind, 'fixed_value': value, 'fit_status': 'optimization_failure',
                'failure': str(exc), 'starts': [], 'best': None, 'near_best_ranges': None}


def explicit_boundaries(data: Data) -> dict[str, dict]:
    return {name: safe_fit(data.y, data.d, kind, fixed)
            for name, kind, fixed in (
                ('sigma_rho_zero', 'rho_fixed', 0.), ('sigma_tau_zero', 'tau_fixed', 0.),
                ('both_variances_zero', 'both_zero', None), ('rank_one_positive', 'r_fixed', 1.),
                ('rank_one_negative', 'r_fixed', -1.))}


def global_reference(unrestricted: dict, boundaries: dict[str, dict]) -> dict[str, Any]:
    required = ('sigma_rho_zero', 'sigma_tau_zero', 'both_variances_zero',
                'rank_one_positive', 'rank_one_negative')
    missing = [name for name in required if boundaries[name]['best'] is None]
    u = unrestricted['best']
    likelihoods = {'unrestricted_loglik': u['independent_loglik'] if u else None,
                   **{name + '_loglik': boundaries[name]['best']['independent_loglik']
                      if boundaries[name]['best'] else None for name in required}}
    if missing or u is None:
        return {**likelihoods, 'reference_consistency_status': 'profile_reference_unresolved',
                'failed_boundaries': missing, 'global_reference_loglik': None,
                'global_reference_model': None, 'boundary_minus_unrestricted_loglik': None}
    candidates = {'unrestricted': u['independent_loglik'],
                  **{name: boundaries[name]['best']['independent_loglik'] for name in required}}
    winner = max(candidates, key=candidates.get)
    boundary_best = max((name for name in required), key=candidates.get)
    difference = candidates[boundary_best] - candidates['unrestricted']
    if difference > NEAR_BEST:
        unrestricted.setdefault('pre_reference_fit_status', unrestricted['fit_status'])
        unrestricted['fit_status'] = 'optimization_unresolved'
    return {**likelihoods,
            'global_reference_loglik': candidates[winner], 'global_reference_model': winner,
            'boundary_minus_unrestricted_loglik': difference,
            'reference_consistency_status': 'optimization_unresolved' if difference > NEAR_BEST else 'PASS',
            'boundary_best_model': boundary_best}


def profile_spec(parameter: str, value: float) -> tuple[str, float]:
    return {'sigma_rho2': 'rho_fixed', 'sigma_tau2': 'tau_fixed',
            'psi': 'psi_fixed', 'r_latent': 'r_fixed'}[parameter], value


def profile_point(data: Data, parameter: str, value: float, reference: float,
                  expansion: int) -> dict[str, Any]:
    kind, fixed = profile_spec(parameter, value)
    fit = safe_fit(data.y, data.d, kind, fixed)
    ll = fit['best']['independent_loglik'] if fit['best'] is not None else None
    delta = 2*(reference-ll) if ll is not None else None
    return {'parameter': parameter, 'profile_region_id': '', 'grid_value': float(value),
            'profile_loglik': ll, 'delta2loglik': delta,
            'in_reference_support_set': delta <= 3.84 if delta is not None else None,
            'fit_status': 'PASS' if ll is not None else 'profile_unresolved',
            'range_expansion_index': expansion, 'endpoint_status': '', 'fit': fit}


def bisect_crossing(data: Data, parameter: str, left: dict, right: dict,
                    reference: float, profile_range: float, expansion: int) -> tuple[list[dict], str]:
    points = []
    lo, hi = left, right
    while hi['grid_value'] - lo['grid_value'] >= 1e-6*profile_range:
        middle = profile_point(data, parameter, (lo['grid_value']+hi['grid_value'])/2, reference, expansion)
        points.append(middle)
        if middle['fit_status'] != 'PASS':
            return points, 'open/unresolved'
        if middle['in_reference_support_set'] == lo['in_reference_support_set']:
            lo = middle
        else:
            hi = middle
    return points, 'bracketed'


def profile_grid(data: Data, parameter: str, reference_info: dict) -> tuple[list[dict], list[dict], dict]:
    if reference_info['reference_consistency_status'] != 'PASS':
        raise ValueError('global profile reference unresolved')
    reference = reference_info['global_reference_loglik']
    sr, st = np.std(data.y, axis=0, ddof=1)
    if parameter == 'sigma_rho2': lo, hi = 0., 4*sr*sr
    elif parameter == 'sigma_tau2': lo, hi = 0., 4*st*st
    elif parameter == 'psi': lo, hi = -4*sr*st, 4*sr*st
    elif parameter == 'r_latent': lo, hi = -1., 1.
    else: raise ValueError(parameter)
    if not hi > lo:
        raise ValueError('nonpositive profile range')
    all_rows = []; left_exp = right_exp = 0
    for expansion in range(17):
        grid = [profile_point(data, parameter, float(x), reference, expansion)
                for x in np.linspace(lo, hi, 161)]
        all_rows.extend(grid)
        if parameter == 'r_latent': break
        grow_left = parameter == 'psi' and bool(grid[0]['in_reference_support_set']) and left_exp < 8
        grow_right = bool(grid[-1]['in_reference_support_set']) and right_exp < 8
        if not (grow_left or grow_right): break
        if grow_left: lo *= 2; left_exp += 1
        if grow_right: hi *= 2; right_exp += 1
    final_range = hi-lo
    # Every final-grid connected support run is retained, including disjoint regions.
    regions = []; ids = ['']*len(grid)
    start = None
    for i in range(len(grid)+1):
        supported = i < len(grid) and bool(grid[i]['in_reference_support_set'])
        if supported and start is None: start = i
        if not supported and start is not None:
            end = i-1; region_id = len(regions)+1
            for j in range(start, end+1): ids[j] = str(region_id)
            regions.append({'profile_region_id': region_id, 'lower': grid[start]['grid_value'],
                            'upper': grid[end]['grid_value'], 'lower_status': '', 'upper_status': ''})
            start = None
    for i, row in enumerate(grid): row['profile_region_id'] = ids[i]
    for region in regions:
        first = next(i for i, v in enumerate(ids) if v == str(region['profile_region_id']))
        last = len(ids)-1-next(i for i, v in enumerate(reversed(ids)) if v == str(region['profile_region_id']))
        for side, inner, outer in [('lower', first, first-1), ('upper', last, last+1)]:
            if outer < 0 or outer >= len(grid):
                if parameter == 'r_latent': status = 'mathematical_domain_boundary'
                elif side == 'lower' and parameter.startswith('sigma_'): status = 'exact_zero_boundary'
                else: status = 'open/unresolved'
            elif grid[outer]['fit_status'] != 'PASS':
                status = 'open/unresolved'
            else:
                left, right = sorted((grid[inner], grid[outer]), key=lambda x: x['grid_value'])
                refined, status = bisect_crossing(data, parameter, left, right, reference, final_range, expansion)
                for point in refined:
                    point['profile_region_id'] = str(region['profile_region_id'])
                all_rows.extend(refined)
                if status == 'bracketed':
                    region[side] = (refined[-1]['grid_value'] if refined else
                                    (left['grid_value']+right['grid_value'])/2)
            if status == 'open/unresolved':
                region[side] = None
            region[side+'_status'] = status
            grid[inner]['endpoint_status'] = status
    for region in regions:
        region['connectivity_status'] = 'resolved'
    for left, right in zip(regions, regions[1:]):
        left_last = max(i for i, value in enumerate(ids) if value == str(left['profile_region_id']))
        right_first = min(i for i, value in enumerate(ids) if value == str(right['profile_region_id']))
        gap = grid[left_last+1:right_first]
        if gap and all(row['fit_status'] == 'profile_unresolved' for row in gap):
            left['connectivity_status'] = right['connectivity_status'] = 'connectivity_unresolved'
    edge_status = {}
    for side, edge in (('lower', grid[0]), ('upper', grid[-1])):
        if edge['fit_status'] == 'profile_unresolved':
            edge_status[side] = 'open/unresolved'
            edge['endpoint_status'] = 'open/unresolved'
        else:
            edge_status[side] = edge['endpoint_status'] or 'outside_support'
    return all_rows, regions, edge_status


def hierarchical_bootstrap(data: Data, count: int = 2_000) -> tuple[list[dict], dict]:
    rng, meta = seeded_rng(HIER_NAMESPACE)
    if meta['seed'] != HIER_SEED:
        raise ValueError('hierarchical seed mismatch')
    rows = []
    for rep in range(1, count+1):
        state_idx = rng.integers(0, len(data.ids), len(data.ids))
        # Fixed order: every A draw in sampled-state order, then every B draw.
        a_idx = rng.integers(0, data.a.shape[1], (len(state_idx), data.a.shape[1]))
        b_idx = rng.integers(0, data.b.shape[1], (len(state_idx), data.b.shape[1]))
        try:
            sampled = summarize_replicas(data.ids[state_idx],
                                         np.take_along_axis(data.a[state_idx], a_idx, axis=1),
                                         np.take_along_axis(data.b[state_idx], b_idx, axis=1))
            fit = safe_fit(sampled.y, sampled.d, full_data=True)
        except (ValueError, OverflowError, FloatingPointError) as exc:
            fit = {'fit_status': 'optimization_failure', 'best': None, 'failure': str(exc)}
        p = fit['best']['scientific_parameters'] if fit['fit_status'] == 'PASS' else {}
        rows.append({'bootstrap_id': rep, **{name: p.get(name) for name in
                    ('sigma_rho2', 'sigma_tau2', 'psi', 'r_latent', 'r_defined', 'boundary_flag')},
                    'fit_status': fit['fit_status']})
    successful = [r for r in rows if r['fit_status'] == 'PASS']
    failures = count-len(successful)
    quantiles = {}
    for name in ('sigma_rho2', 'sigma_tau2', 'psi', 'r_latent'):
        values = [r[name] for r in successful if r[name] is not None]
        quantiles[name] = np.quantile(values, QUANTILES, method='linear').tolist() if values else None
    summary = {'rng': meta, 'attempted_count': count, 'successful_fits': len(successful),
               'optimization_failures': sum(r['fit_status'] == 'optimization_failure' for r in rows),
               'optimization_unstable': sum(r['fit_status'] == 'optimization_unstable' for r in rows),
               'boundary_fits': sum(bool(r['boundary_flag']) for r in successful),
               'success_fraction': len(successful)/count,
               'undefined_correlation_fraction': sum(r['r_defined'] is False for r in successful)/len(successful) if successful else None,
               'percentiles_conditional_on_successful_fits': quantiles,
               'interpretation': 'hierarchical-bootstrap EIV sensitivity numerically unreliable' if failures/count > 0.05 else 'conditional on successful fits'}
    return rows, summary


def pooled_noise_fit(data: Data) -> dict:
    pooled_a = float(np.sum((data.a.shape[1]-1)*np.var(data.a, axis=1, ddof=1)) /
                     (len(data.ids)*(data.a.shape[1]-1)))
    pooled_b = float(np.sum((data.b.shape[1]-1)*np.var(data.b, axis=1, ddof=1)) /
                     (len(data.ids)*(data.b.shape[1]-1)))
    d = np.tile([pooled_a/data.a.shape[1], pooled_b/data.b.shape[1]], (len(data.ids), 1))
    return {'pooled_A_variance': pooled_a, 'pooled_B_variance': pooled_b,
            'fit': safe_fit(data.y, d, full_data=True)}


def leave_one_out(data: Data, full_fit: dict) -> tuple[list[dict], dict]:
    rows = []
    for i, sid in enumerate(data.ids):
        fit = safe_fit(np.delete(data.y, i, axis=0), np.delete(data.d, i, axis=0), full_data=True)
        p = fit['best']['scientific_parameters'] if fit['fit_status'] == 'PASS' else {}
        rows.append({'excluded_base_run_id': int(sid), **{name: p.get(name) for name in
                    ('mu_rho', 'mu_tau', 'sigma_rho2', 'sigma_tau2', 'psi', 'r_latent', 'r_defined')},
                    'fit_status': fit['fit_status']})
    full = full_fit['best']['scientific_parameters'] if full_fit['fit_status'] == 'PASS' else {}
    summary = {'full_data': full, 'optimization_failure_count': sum(r['fit_status'] == 'optimization_failure' for r in rows),
               'optimization_unstable_count': sum(r['fit_status'] == 'optimization_unstable' for r in rows),
               'undefined_correlation_count': sum(r['r_defined'] is False for r in rows),
               'psi_sign_changes': sum(r['psi']*full['psi'] < 0 for r in rows
                                       if r['psi'] is not None and full.get('psi') is not None)}
    for name in ('mu_rho', 'mu_tau', 'sigma_rho2', 'sigma_tau2', 'psi', 'r_latent'):
        values = [r[name] for r in rows if r[name] is not None]
        summary[name+'_min_median_max'] = [min(values), float(np.median(values)), max(values)] if values else None
    return rows, summary


def residual_diagnostics(data: Data) -> dict[str, Any]:
    result = {}
    for label, block in (('A', data.a), ('B', data.b)):
        residual = (block-block.mean(axis=1, keepdims=True)).ravel()
        sds = np.std(block, axis=1, ddof=1)
        result[label] = {'pooled_centered_residual': {
            'mean': float(np.mean(residual)), 'sd': float(np.std(residual, ddof=1)),
            'skewness': float(skew(residual, bias=False)),
            'excess_kurtosis': float(kurtosis(residual, fisher=True, bias=False)),
            'quantiles': dict(zip(('q01', 'q05', 'q50', 'q95', 'q99'),
                                  np.quantile(residual, [.01, .05, .5, .95, .99], method='linear').tolist()))},
            'state_within_sd': dict(zip((str(sid) for sid in data.ids), sds.tolist()))}
    return result


def scenarios() -> dict[str, np.ndarray]:
    result = {'S0': np.zeros((2, 2)), 'S1': np.diag([0., 9.107e-6]),
              'S2': np.diag([5e-6, 0.]), 'S3': np.diag([1e-6, 2.5e-6])}
    variance_pairs = ((2.5e-6, 5e-6), (5e-6, 9.107e-6), (1e-5, 2.5e-5),
                      (2.5e-6, 2.5e-5), (1e-5, 5e-6))
    for j, (vr, vt) in enumerate(variance_pairs, 4):
        for suffix, r in (('neg', -.3), ('zero', 0.), ('pos', .3)):
            p = r*math.sqrt(vr*vt)
            result[f'S{j}_{suffix}'] = np.array([[vr, p], [p, vt]])
    return result


def candidate_designs() -> list[tuple[int, int, int]]:
    return list(product((36, 48, 60, 72), (15, 30, 60), (10, 15, 20)))


def incremental_cost(n: int, ka: int, kb: int) -> int:
    return 36*(8*(ka-15)+32*(kb-10))+(n-36)*(14+8*ka+32*kb)


def psd_symmetric_root(cov: np.ndarray) -> np.ndarray:
    eig, vectors = np.linalg.eigh(cov)
    threshold = -1e-12*max(1., float(max(eig)))
    if np.any(eig < threshold):
        raise ValueError('PSD eigenvalue below frozen roundoff tolerance')
    return (vectors*np.sqrt(np.maximum(eig, 0.))) @ vectors.T


def maximum_realization(data: Data, scenario_id: str, latent_family: str,
                        measurement_engine: str, replica_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    rng, meta = seeded_rng(PRECISION_NAMESPACE, scenario_id, latent_family,
                           measurement_engine, replica_id)
    cov = scenarios()[scenario_id]
    scale = cov if latent_family == 'L1' else 3*cov/5
    root = psd_symmetric_root(scale)
    # A common mean is immaterial to covariance, jackknife width and coverage.
    latent = rng.standard_normal((72, 2)) @ root.T
    if latent_family == 'L2':
        latent /= np.sqrt(rng.chisquare(5, size=72)/5)[:, None]
    elif latent_family != 'L1':
        raise ValueError('unknown latent family')
    templates = rng.integers(0, len(data.ids), 72)
    if measurement_engine == 'M1':
        a_sd = np.std(data.a, axis=1, ddof=1)[templates]
        b_sd = np.std(data.b, axis=1, ddof=1)[templates]
        a_residual = rng.normal(size=(72, 60))*a_sd[:, None]
        b_residual = rng.normal(size=(72, 20))*b_sd[:, None]
    elif measurement_engine == 'M2':
        a_pool = data.a-data.a.mean(axis=1, keepdims=True)
        b_pool = data.b-data.b.mean(axis=1, keepdims=True)
        a_indices = rng.integers(0, data.a.shape[1], (72, 60))
        b_indices = rng.integers(0, data.b.shape[1], (72, 20))
        a_residual = np.take_along_axis(a_pool[templates], a_indices, axis=1)
        b_residual = np.take_along_axis(b_pool[templates], b_indices, axis=1)
    else:
        raise ValueError('unknown measurement engine')
    return latent, a_residual, b_residual, templates, meta


def candidate_means(realization: tuple, n: int, ka: int, kb: int) -> np.ndarray:
    latent, a_residual, b_residual, _, _ = realization
    return np.column_stack((latent[:n, 0] + a_residual[:n, :ka].mean(axis=1),
                            latent[:n, 1] + b_residual[:n, :kb].mean(axis=1)))


def covariance_and_jackknife(y: np.ndarray) -> tuple[float, float, tuple[float, float]]:
    n = len(y)
    if n < 3 or not np.isfinite(y).all():
        raise ValueError('jackknife requires at least 3 finite states')
    x, z = y[:, 0], y[:, 1]
    psi = float(np.cov(x, z, ddof=1)[0, 1])
    sum_x, sum_z, sum_xz = np.sum(x), np.sum(z), np.dot(x, z)
    m = n-1
    omitted = ((sum_xz-x*z) - (sum_x-x)*(sum_z-z)/m)/(m-1)
    se = float(np.sqrt((n-1)/n*np.sum((omitted-omitted.mean())**2)))
    if not np.isfinite(psi) or not np.isfinite(se):
        raise ValueError('non-finite covariance or jackknife SE')
    return psi, se, (psi-1.96*se, psi+1.96*se)


def precision_simulation(data: Data, count: int = 3_000) -> tuple[list[dict], dict]:
    designs = candidate_designs(); rows = []; rng_records = []
    for scenario_id, cov in scenarios().items():
        truth = float(cov[0, 1])
        for family, engine in product(('L1', 'L2'), ('M1', 'M2')):
            cells = {design: {'psi': [], 'width': [], 'coverage': [], 'failures': {}} for design in designs}
            for rep in range(1, count+1):
                try:
                    realization = maximum_realization(data, scenario_id, family, engine, rep)
                    rng_records.append(realization[4])
                except (ValueError, FloatingPointError, OverflowError) as exc:
                    for cell in cells.values():
                        kind = type(exc).__name__ + ': ' + str(exc)
                        cell['failures'][kind] = cell['failures'].get(kind, 0)+1
                    continue
                for design, cell in cells.items():
                    try:
                        y = candidate_means(realization, *design)
                        estimate, _, interval = covariance_and_jackknife(y)
                        cell['psi'].append(estimate)
                        cell['width'].append(interval[1]-interval[0])
                        cell['coverage'].append(interval[0] <= truth <= interval[1])
                    except (ValueError, FloatingPointError, OverflowError) as exc:
                        kind = type(exc).__name__ + ': ' + str(exc)
                        cell['failures'][kind] = cell['failures'].get(kind, 0)+1
            group = []
            for (n, ka, kb), cell in cells.items():
                estimates = np.asarray(cell['psi']); widths = np.asarray(cell['width'])
                valid = len(estimates); failed = count-valid
                mean = float(np.mean(estimates)) if valid else None
                median_width = float(np.median(widths)) if valid else None
                group.append({'scenario_id': scenario_id, 'latent_family': family, 'measurement_engine': engine,
                              'N': n, 'K_A': ka, 'K_B': kb, 'attempted_count': count,
                              'valid_count': valid, 'failed_count': failed,
                              'cell_status': 'complete' if failed == 0 else 'incomplete',
                              'failure_types': json.dumps(cell['failures'], sort_keys=True),
                              'incremental_cost': incremental_cost(n, ka, kb), 'true_psi': truth,
                              'mean_psi_hat': mean, 'bias': mean-truth if valid else None,
                              'mc_sd': float(np.std(estimates, ddof=1)) if valid >= 2 else None,
                              'rmse': float(np.sqrt(np.mean((estimates-truth)**2))) if valid else None,
                              'median_interval_width': median_width,
                              'q90_interval_width': float(np.quantile(widths, .9, method='linear')) if valid else None,
                              'coverage': float(np.mean(cell['coverage'])) if valid else None})
            baseline = next(row for row in group if (row['N'], row['K_A'], row['K_B']) == (36, 15, 10))
            for row in group:
                if baseline['median_interval_width'] in (None, 0) or row['median_interval_width'] is None:
                    gain = None
                else:
                    gain = 1-row['median_interval_width']/baseline['median_interval_width']
                row['precision_gain'] = gain
                row['precision_gain_per_1000'] = 'NA' if row['incremental_cost'] == 0 else (
                    gain/(row['incremental_cost']/1000) if gain is not None else None)
                if row['cell_status'] != 'complete':
                    row['pareto_dominated'] = 'NA'
                else:
                    row['pareto_dominated'] = any(
                        other is not row and other['cell_status'] == 'complete' and
                        other['incremental_cost'] <= row['incremental_cost'] and
                        other['median_interval_width'] <= row['median_interval_width'] and
                        (other['incremental_cost'] < row['incremental_cost'] or
                         other['median_interval_width'] < row['median_interval_width'])
                        for other in group)
            rows.extend(group)
    return rows, {'master_namespace': PRECISION_NAMESPACE, 'rng_realizations': rng_records,
                  'numpy_version': np.__version__, 'monte_carlo_replicates_per_cell': count}


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = SCHEMAS[path.name]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in columns})


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(json_safe(obj), indent=2, allow_nan=False, sort_keys=True)+'\n')


def results_markdown(decomposition_rows: list[dict], fit: dict, reference: dict,
                     regions: dict, bootstrap_summary: dict, precision_rows: list[dict],
                     outer_meta: dict | None = None, profile_edges: dict | None = None) -> str:
    # Reporting is deliberately bounded to claims authorized by Sections 3, 7, 15, 30-34.
    def fmt(value: Any) -> str:
        if value is None:
            return 'undetermined'
        return f'{value:.9g}' if isinstance(value, (int, float, np.number)) else str(value)
    lines = ['# Post-Phase-5 EIV results', '',
             '**post-outcome exploratory**', '',
             'The frozen Phase 5A and 5B results in `RESULTS_PHASE5.md` are unchanged.', '',
             '## Distribution-light decomposition', '', '**post-outcome exploratory**', '',
             'Raw negative variance components mean positive heterogeneity was unresolved by this estimator, not negative physical variance or exact homogeneity.', '',
             'Outer-only percentiles are sensitivity ranges, not confidence intervals. Their resampling retains replica noise already embedded in the observed means; latent-variance coverage is unestablished.', '',
             '## Conditional plug-in EIV', '', '**post-outcome exploratory**', '',
             f"Full-data unrestricted fit status: `{fit['fit_status']}`.",
             f"Profile-reference status: `{reference['reference_consistency_status']}`.", '',
             '95%-reference likelihood support sets are descriptive. Boundary calibration and nominal 95% frequentist coverage are not established.', '',
             'If a marginal latent-variance support set includes zero, latent correlation is weakly identified and undefined at the zero-variance boundary.', '',
             '## Measurement variance and robustness', '', '**post-outcome exploratory**', '',
             f"Hierarchical-bootstrap status: {bootstrap_summary['interpretation']}.", '',
             'Successful-fit percentiles are conditional on successful fits and make no calibrated coverage claim.', '',
             '## Precision planning', '', '**post-outcome exploratory**', '',
             'Median width ranks this planning metric only; where empirical coverage differs, a narrower interval does not establish superior interval performance.', '',
             'Incomplete cells do not enter Pareto ranking. Frontier position does not select a Phase 6 route.', '',
             f"Simulation cells: {len(precision_rows)}.", '',
             'See the CSV and JSON artifacts for every cell, failure, and unresolved support point.', '']
    if outer_meta is not None:
        lines.insert(lines.index('## Conditional plug-in EIV'),
            f"Outer-only bootstrap: {outer_meta['valid_count']}/{outer_meta['attempted_count']} valid; {outer_meta['failed_count']} failed attempts (IDs and failure details in `post_phase5_eiv_fit.json`).")
    if reference['reference_consistency_status'] == 'optimization_unresolved':
        lines.insert(lines.index('## Measurement variance and robustness'),
            f"Unrestricted estimate suppressed: boundary `{reference['boundary_best_model']}` exceeds it by {fmt(reference['boundary_minus_unrestricted_loglik'])} log-likelihood units; Tier-2 profile support sets are not interpreted.")
    for row in decomposition_rows:
        lines.insert(lines.index('## Conditional plug-in EIV'),
            f"- {row['parameter']}: observed variance {fmt(row['observed_variance'])}; mean measurement variance {fmt(row['mean_measurement_variance'])}; raw deconvolved variance {fmt(row['raw_deconvolved_variance'])}; raw signal fraction {fmt(row['raw_signal_fraction'])}; outer-only q025/q500/q975 {fmt(row['outer_q025'])}/{fmt(row['outer_q500'])}/{fmt(row['outer_q975'])}.")
    if fit['fit_status'] == 'PASS' and reference['reference_consistency_status'] == 'PASS':
        p = fit['best']['scientific_parameters']
        zero_supported = any(
            r['lower'] == 0 and r['lower_status'] == 'exact_zero_boundary'
            for name in ('sigma_rho2', 'sigma_tau2') for r in regions.get(name, []))
        if zero_supported:
            correlation = 'weakly identified; undefined at the zero-variance boundary'
        elif p['r_defined']:
            correlation = str(p['r_latent'])
        else:
            correlation = 'undefined'
        lines.insert(lines.index('## Measurement variance and robustness'),
            f"Conditional plug-in estimates: sigma_rho2={p['sigma_rho2']:.9g}, sigma_tau2={p['sigma_tau2']:.9g}, psi={p['psi']:.9g}; latent correlation: {correlation}.")
    for parameter, parameter_regions in regions.items():
        rendered = '; '.join(f"[{fmt(r['lower'])}, {fmt(r['upper'])}] ({r['lower_status']}, {r['upper_status']}; {r['connectivity_status']})"
                             for r in parameter_regions) or 'no resolved supported region'
        if profile_edges and parameter in profile_edges:
            rendered += f"; grid edges: {profile_edges[parameter]}"
        lines.insert(lines.index('## Measurement variance and robustness'),
                     f'{parameter} 95%-reference support: {rendered}.')
    lines.insert(lines.index('## Precision planning'),
        f"Hierarchical bootstrap: {bootstrap_summary['successful_fits']}/{bootstrap_summary['attempted_count']} successful; {bootstrap_summary['optimization_failures']} optimization failures; {bootstrap_summary['optimization_unstable']} unstable; {bootstrap_summary['boundary_fits']} boundary fits; undefined-correlation fraction {bootstrap_summary['undefined_correlation_fraction']}.")
    complete = sum(row['cell_status'] == 'complete' for row in precision_rows)
    frontier = sum(row['pareto_dominated'] is False for row in precision_rows)
    lines.insert(lines.index('See the CSV and JSON artifacts for every cell, failure, and unresolved support point.'),
                 f'Complete simulation cells: {complete}/{len(precision_rows)}; non-dominated complete cells: {frontier}.')
    return '\n'.join(lines)


def phase6_package(decomposition_rows: list[dict], outer_meta: dict, fit: dict,
                   reference: dict, regions: dict, profile_edges: dict,
                   bootstrap_summary: dict, pooled_fit: dict, loo_summary: dict,
                   residual_diag: dict, precision_rows: list[dict]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return 'undetermined'
        return f'{value:.6g}' if isinstance(value, (int, float, np.number)) else str(value)

    def span(rows: list[dict], key: str) -> str:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float, np.number))]
        return f'{fmt(min(values))}–{fmt(max(values))}' if values else 'NA'

    lines = ['# Phase 6 evidence package', '',
             '## What frozen Phase 5 established', '',
             'Frozen Phase 5A: signature supported; mu_rho = 0.0484693, 95% interval [0.0461314, 0.0509002].',
             'Frozen Phase 5B: primary association insufficiently precise; psi = -6.9500e-6, 95% percentile interval [-2.5197e-5, 1.0202e-5]. These primary results remain unchanged.', '',
             '## Identifiability of latent quantities', '', '**post-outcome exploratory**', '',
             f"Unrestricted fit status: `{fit['fit_status']}`; global reference status: `{reference['reference_consistency_status']}`."]
    if reference['reference_consistency_status'] == 'optimization_unresolved':
        lines.append(f"Boundary `{reference['boundary_best_model']}` exceeded the unrestricted likelihood by {fmt(reference['boundary_minus_unrestricted_loglik'])}; its unrestricted point estimate and profile support sets are not interpreted.")
    elif fit['fit_status'] == 'PASS' and reference['reference_consistency_status'] == 'PASS':
        p = fit['best']['scientific_parameters']
        lines.append(f"Conditional plug-in estimates: sigma_rho2={fmt(p['sigma_rho2'])}, sigma_tau2={fmt(p['sigma_tau2'])}, psi={fmt(p['psi'])}.")
    for name in ('sigma_rho2', 'sigma_tau2', 'psi', 'r_latent'):
        if name in regions:
            support = '; '.join(f"[{fmt(r['lower'])}, {fmt(r['upper'])}] ({r['lower_status']}, {r['upper_status']}; {r['connectivity_status']})"
                                for r in regions[name]) or 'no resolved supported region'
            lines.append(f'{name} descriptive 95%-reference support: {support}; grid edges {profile_edges.get(name, {})}.')
    zero_supported = any(r['lower'] == 0 and r['lower_status'] == 'exact_zero_boundary'
                         for name in ('sigma_rho2', 'sigma_tau2') for r in regions.get(name, []))
    if zero_supported:
        lines.append('Latent correlation is weakly identified and undefined at a supported zero-variance boundary; its numerical point is not emphasized.')
    lines.extend(['These are descriptive likelihood support sets; boundary calibration and nominal 95% coverage are not established.', '',
                  '## Variance and covariance uncertainty', '', '**post-outcome exploratory**', '',
                  f"Outer-only bootstrap: {outer_meta['valid_count']}/{outer_meta['attempted_count']} valid, {outer_meta['failed_count']} failed; ranges are sensitivity ranges, not confidence intervals."])
    for row in decomposition_rows:
        lines.append(f"{row['parameter']}: observed variance {fmt(row['observed_variance'])}; mean measurement variance {fmt(row['mean_measurement_variance'])}; raw deconvolved variance {fmt(row['raw_deconvolved_variance'])}; raw signal fraction {fmt(row['raw_signal_fraction'])}; outer-only q025/q500/q975 {fmt(row['outer_q025'])}/{fmt(row['outer_q500'])}/{fmt(row['outer_q975'])}.")
    lines.append(f"Hierarchical EIV bootstrap: {bootstrap_summary['successful_fits']}/{bootstrap_summary['attempted_count']} successful; {bootstrap_summary['optimization_failures']} optimization failures; {bootstrap_summary['optimization_unstable']} unstable; {bootstrap_summary['boundary_fits']} boundary fits; status: {bootstrap_summary['interpretation']}.")
    for name, quantiles in bootstrap_summary['percentiles_conditional_on_successful_fits'].items():
        lines.append(f'{name} successful-fit q025/q500/q975: {"/".join(fmt(v) for v in quantiles) if quantiles is not None else "NA"}.')
    lines.extend(['These percentiles are conditional on successful fits and have no calibrated coverage claim.', '',
                  '## Model sensitivity', '', '**post-outcome exploratory**', ''])
    pooled = pooled_fit['fit']
    lines.append(f"Pooled A/B within-state variances: {fmt(pooled_fit['pooled_A_variance'])}/{fmt(pooled_fit['pooled_B_variance'])}; pooled-noise fit status: `{pooled['fit_status']}`.")
    if pooled['fit_status'] == 'PASS':
        p = pooled['best']['scientific_parameters']
        lines.append(f"Pooled-noise estimates: sigma_rho2={fmt(p['sigma_rho2'])}, sigma_tau2={fmt(p['sigma_tau2'])}, psi={fmt(p['psi'])}.")
    lines.append(f"Leave-one-state-out: {loo_summary['optimization_failure_count']} failed, {loo_summary['optimization_unstable_count']} unstable, {loo_summary['undefined_correlation_count']} undefined correlations, {loo_summary['psi_sign_changes']} psi sign reversals.")
    for name in ('mu_rho', 'mu_tau', 'sigma_rho2', 'sigma_tau2', 'psi', 'r_latent'):
        lines.append(f"{name} full={fmt(loo_summary['full_data'].get(name))}; LOO min/median/max={','.join(fmt(v) for v in loo_summary[name+'_min_median_max']) if loo_summary[name+'_min_median_max'] is not None else 'NA'}.")
    for block in ('A', 'B'):
        residual = residual_diag[block]['pooled_centered_residual']
        lines.append(f"{block} centered residuals: SD={fmt(residual['sd'])}, skewness={fmt(residual['skewness'])}, excess kurtosis={fmt(residual['excess_kurtosis'])}, q01/q99={fmt(residual['quantiles']['q01'])}/{fmt(residual['quantiles']['q99'])}.")
    lines.extend(['## Precision and cost', '', '**post-outcome exploratory**', '',
                  f"Complete cells: {sum(row['cell_status'] == 'complete' for row in precision_rows)}/{len(precision_rows)}. Incomplete cells remain descriptive and do not enter Pareto ranking.",
                  'The table summarizes complete cells by scenario, latent family, and measurement engine; the full candidate metrics are in `post_phase5_precision.csv`.', '',
                  '| Scenario | Family | Engine | Complete | True psi | Baseline median width | Median width range | Coverage range | MC SD range | RMSE range | Q90 width range | Cost range | Frontier count |',
                  '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    groups: dict[tuple, list[dict]] = {}
    for row in precision_rows:
        groups.setdefault((row['scenario_id'], row['latent_family'], row['measurement_engine']), []).append(row)
    for (scenario, family, engine), group in groups.items():
        complete = [row for row in group if row['cell_status'] == 'complete']
        baseline = next((row for row in complete if (row['N'], row['K_A'], row['K_B']) == (36, 15, 10)), None)
        lines.append('| ' + ' | '.join((scenario, family, engine, str(len(complete)),
            fmt(group[0]['true_psi']), fmt(baseline['median_interval_width']) if baseline else 'NA',
            span(complete, 'median_interval_width'), span(complete, 'coverage'),
            span(complete, 'mc_sd'), span(complete, 'rmse'),
            span(complete, 'q90_interval_width'), span(complete, 'incremental_cost'),
            str(sum(row['pareto_dominated'] is False for row in complete)))) + ' |')
    unresolved_sides = sum(status == 'open/unresolved' for edges in profile_edges.values()
                           for status in edges.values())
    incomplete_cells = sum(row['cell_status'] != 'complete' for row in precision_rows)
    lines.extend(['', 'Median width ranks this planning metric only; differing coverage prevents a narrower interval from establishing superior interval performance. No Phase 6 route follows automatically from the frontier.', '',
                  '## Unresolved scientific questions', '',
                  f'{unresolved_sides} profile grid sides remain open/unresolved; {incomplete_cells} precision cells are incomplete; {bootstrap_summary["optimization_failures"] + bootstrap_summary["optimization_unstable"]} hierarchical fits failed or were unstable.',
                  'Latent heterogeneity, covariance precision, Gaussian-model adequacy, and the limits of empirical residual support remain conditional on the reported ranges, failures, and scenarios. A later design process must decide the next scientific question.', ''])
    return '\n'.join(lines)


def formal_analysis(output_dir: Path) -> None:
    data = load_inputs()  # All hash and input-contract checks precede any analysis.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('output directory must be absent or empty')
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'protocol_id': 'Post-Phase5-EIV-v3.1', 'protocol_sha256': sha256(PROTOCOL),
                'python_version': platform.python_version(), 'numpy_version': np.__version__,
                'scipy_version': scipy.__version__, 'analysis_script_sha256': sha256(Path(__file__)),
                'replica_input_path': str(REPLICA_PATH), 'replica_input_sha256': sha256(REPLICA_PATH),
                'state_input_path': str(STATE_PATH), 'state_input_sha256': sha256(STATE_PATH)}
    write_json(output_dir/'post_phase5_eiv_manifest.json', manifest)
    decomposition_rows, outer_meta = outer_only_bootstrap(data)
    write_csv(output_dir/'post_phase5_eiv_decomposition.csv', decomposition_rows)
    fit = safe_fit(data.y, data.d, full_data=True)
    boundaries = explicit_boundaries(data)
    reference = global_reference(fit, boundaries)
    fit_report = {'unrestricted': fit, 'explicit_boundaries': boundaries, 'global_reference': reference,
                  **reference,
                  'optimizer': {'method': 'L-BFGS-B', **OPT_OPTIONS}, 'outer_bootstrap': outer_meta}
    support_rows = []; regions = {}; profile_edges = {}
    if reference['reference_consistency_status'] == 'PASS' and fit['fit_status'] == 'PASS':
        for parameter in ('sigma_rho2', 'sigma_tau2', 'psi', 'r_latent'):
            rows, regions[parameter], profile_edges[parameter] = profile_grid(data, parameter, reference)
            # Constrained-fit starts and returned vectors remain inspectable in JSON.
            fit_report.setdefault('profile_fits', {})[parameter] = [r['fit'] for r in rows]
            support_rows.extend(rows)
    else:
        fit_report['profile_interpretation'] = 'stopped: full-data or global reference unresolved'
    fit_report['profile_edges'] = profile_edges
    write_json(output_dir/'post_phase5_eiv_fit.json', fit_report)
    write_csv(output_dir/'post_phase5_eiv_support.csv', support_rows)
    bootstrap_rows, bootstrap_summary = hierarchical_bootstrap(data)
    write_csv(output_dir/'post_phase5_eiv_bootstrap.csv', bootstrap_rows)
    write_json(output_dir/'post_phase5_eiv_bootstrap_summary.json', bootstrap_summary)
    pooled_fit = pooled_noise_fit(data)
    write_json(output_dir/'post_phase5_eiv_pooled_noise.json', pooled_fit)
    loo_rows, loo_summary = leave_one_out(data, fit)
    write_csv(output_dir/'post_phase5_eiv_loo.csv', loo_rows)
    write_json(output_dir/'post_phase5_eiv_loo_summary.json', loo_summary)
    residual_diag = residual_diagnostics(data)
    write_json(output_dir/'post_phase5_residual_diagnostics.json', residual_diag)
    precision_rows, precision_meta = precision_simulation(data)
    write_csv(output_dir/'post_phase5_precision.csv', precision_rows)
    write_json(output_dir/'post_phase5_precision_rng.json', precision_meta)
    (output_dir/'POST_PHASE5_EIV_RESULTS.md').write_text(
        results_markdown(decomposition_rows, fit, reference, regions, bootstrap_summary,
                         precision_rows, outer_meta, profile_edges))
    (output_dir/'PHASE6_EVIDENCE_PACKAGE.md').write_text(
        phase6_package(decomposition_rows, outer_meta, fit, reference, regions, profile_edges,
                       bootstrap_summary, pooled_fit, loo_summary, residual_diag, precision_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    formal_analysis(args.output_dir)


if __name__ == '__main__':
    main()
