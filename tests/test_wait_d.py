"""Small synthetic fixtures only: never train any real Phase 5 checkpoint."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from reflexml.branching import order_hashes, states_equal
from reflexml.checkpoint import load_checkpoint, save_checkpoint
from reflexml.config import ExperimentConfig
from reflexml.data import make_loaders_from_datasets
from reflexml.model import FashionMLP
from reflexml.wait_d import (ARMS, analyze_records, compare_reproduction, run_unit,
                            schedules, summarize, validate_records, write_json, validate_source_state, inspect_run)


def gaps(p=.1, q=.05):
    values = np.empty((12, 2, 4, 2))
    values[..., 0], values[..., 1] = p, q
    return values


class AnalysisTests(unittest.TestCase):
    def test_formulas(self):
        row = summarize(gaps())['delays'][1]
        self.assertAlmostEqual(row['P'], .1)
        self.assertAlmostEqual(row['Q'], .05)
        self.assertAlmostEqual(row['A'], -.05)
        self.assertAlmostEqual(row['C'], .5)

    def test_ratio_of_means_and_sample_se(self):
        x = gaps()
        x[..., 0] = np.arange(1, 13)[:, None, None] * .1
        x[..., 1] = .1
        row = summarize(x)['delays'][1]
        self.assertAlmostEqual(row['C'], 1 - .1 / .65)
        self.assertNotAlmostEqual(row['C'], np.mean(1 - .1 / x[:, 0, 0, 0]))
        self.assertAlmostEqual(row['SE_state'], np.std(np.arange(1, 13)*.1, ddof=1)/np.sqrt(12))

    def test_denominator_absolute_strict_boundary(self):
        for p, expected in ((.01, False), (.010001, True), (0, False), (-.1, False)):
            with self.subTest(p=p):
                row = summarize(gaps(p, .005))['delays'][1]
                self.assertEqual(row['denominator_valid'], expected)
                self.assertEqual(row['C'] is not None, expected)

    def test_denominator_se_strict_boundary(self):
        deviations = np.arange(12, dtype=float) - 5.5
        se = deviations.std(ddof=1)/np.sqrt(12)
        x = gaps()
        x[..., 0] = (deviations + 2*se)[:, None, None]
        row = summarize(x)['delays'][1]
        self.assertFalse(row['denominator_valid'])

    def test_denominator_split_sign(self):
        x = gaps(); x[:, 0, :, 0] = -.01; x[:, 1, :, 0] = .3
        self.assertFalse(summarize(x)['delays'][1]['denominator_valid'])

    def test_movement_and_overshoot(self):
        x = gaps(); x[:3, :, :, 1] = -.02; x[3:6, :, :, 1] = -.11
        row = summarize(x)['delays'][1]
        self.assertEqual(row['n'], 9)
        self.assertEqual(sum(row['overshoot_i']), 6)

    def test_movement_strict_and_positive(self):
        self.assertEqual(summarize(gaps(.1, -.1))['delays'][1]['n'], 0)
        self.assertEqual(summarize(gaps(-.1, 0))['delays'][1]['n'], 0)

    def test_substantial_boundary(self):
        # Exact binary values avoid decimal roundoff at the frozen boundary.
        x = gaps(.125, .09375)
        self.assertEqual(summarize(x)['T'], [1, 2, 3, 4])
        x[:4, :, :, 1] = .125
        self.assertEqual(summarize(x)['T'], [])

    def test_split_conflict_case_a(self):
        x = gaps(); x[:, 0, :, 0] = 0
        self.assertTrue(summarize(x)['delays'][1]['split_conflict'])

    def test_split_conflict_case_b(self):
        x = gaps(); x[:, 0, :, 1] = .2
        self.assertTrue(summarize(x)['delays'][1]['split_conflict'])

    def test_go(self):
        result = summarize(gaps())
        self.assertEqual(result['classification'], 'GO')
        self.assertEqual(result['secondary']['label'], 'BROADLY_SIMILAR_CLOSURE')

    def test_go_coarse_summary(self):
        x = gaps(.125, .0625); x[:, :, 0, 1] = 0
        self.assertEqual(summarize(x)['secondary']['label'], 'COARSE_DELAY_VARIATION')

    def test_wait3_specific(self):
        x = gaps(.1, .09); x[:, :, 2, 1] = .05
        self.assertEqual(summarize(x)['classification'], 'NO_GO_WAIT3_SPECIFIC')

    def test_no_non_d3_closure(self):
        self.assertEqual(summarize(gaps(.1, .11))['classification'], 'NO_GO_NO_NON_D3_CLOSURE')

    def test_overlap_precedence(self):
        x = gaps(.1, .11); x[:, :, 2, 1] = .05
        result = summarize(x)
        self.assertTrue(result['predicates']['NO_GO_WAIT3_SPECIFIC'])
        self.assertEqual(result['classification'], 'NO_GO_NO_NON_D3_CLOSURE')

    def test_split_conflict_blocks_no_go(self):
        x = gaps(.1, .09); x[:, :, 2, 1] = .05; x[:, 0, 0, 1] = .11
        self.assertEqual(summarize(x)['classification'], 'AMBIGUOUS')

    def test_path2_ignores_d3_split_conflict(self):
        x = gaps(.1, .11); x[:, 0, 2, 0] = 0
        self.assertEqual(summarize(x)['classification'], 'NO_GO_NO_NON_D3_CLOSURE')

    def test_ambiguous(self):
        x = gaps(.1, .09); x[:, :, :2, 1] = .05
        self.assertEqual(summarize(x)['classification'], 'AMBIGUOUS')
        self.assertIsNone(summarize(x)['secondary'])

    def test_no_clipping(self):
        self.assertGreater(summarize(gaps(.1, -.02))['delays'][1]['C'], 1)

    def test_bad_shape_nonfinite(self):
        for x in (np.ones((11, 2, 4, 2)), gaps(np.nan)):
            with self.assertRaises(ValueError): summarize(x)


class KernelAndValidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.config = ExperimentConfig(train_size=8, val_size=4, batch_size=4, hidden_size=4)
        cls.dataset = TensorDataset(torch.rand(12, 1, 28, 28), torch.arange(12) % 10)
        data = make_loaders_from_datasets(cls.dataset, [], cls.config, audit_order=True)
        model = FashionMLP(4)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        nn.CrossEntropyLoss()(model(cls.dataset.tensors[0][:4]), cls.dataset.tensors[1][:4]).backward()
        optimizer.step()
        with TemporaryDirectory() as temp:
            path = Path(temp)/'source.pt'
            save_checkpoint(path, 14, model, optimizer, cls.config, data.train_generator, data.split_fingerprint)
            cls.checkpoint = load_checkpoint(path, torch.device('cpu'))
        cls.original = deepcopy(cls.checkpoint)
        cls.results = run_unit(cls.checkpoint, cls.dataset, 100)
        cls.historical = {}
        for arm, prefix in (('Now', 'now'), ('Wait3', 'wait')):
            row = cls.results[arm]
            cls.historical.update({prefix+'_history': row['history'][:4],
                prefix+'_realized_orders': row['orders'][:4],
                prefix+'_epoch_order_sha256': order_hashes(row['orders'][:4]),
                prefix+'_state_finiteness_evidence': row['state_finiteness_evidence'][:4]})

    def record(self, arm):
        result = self.results[arm]
        return {'state': 1, 'future': 1, 'arm': arm, 'status': 'SUCCESS',
                'binding': {'config': self.config.to_dict(), 'population_size': 12},
                'history': deepcopy(result['history']), 'orders': deepcopy(result['orders']),
                'order_sha256': order_hashes(result['orders']),
                'evidence': deepcopy(result['state_finiteness_evidence'])}

    def roster(self):
        records, bindings, historical = [], {}, {}
        for s in range(1, 13):
            for r in (1, 2):
                bindings[f'{s:03d}-A{r}'] = self.record('Now')['binding']
                historical[s, r] = self.historical
                for arm in ARMS:
                    row = self.record(arm); row.update(state=s, future=r); records.append(row)
        return records, bindings, historical

    def test_delay_schedules_and_horizons(self):
        for arm, (_, horizon) in ARMS.items():
            self.assertEqual(len(self.results[arm]['history']), horizon)
            self.assertEqual([r['learning_rate'] for r in self.results[arm]['history']], schedules()[arm])

    def test_overlapping_orders(self):
        for arm, (_, horizon) in ARMS.items():
            self.assertEqual(self.results[arm]['orders'], self.results['Now']['orders'][:horizon])

    def test_deterministic_h5_and_now_wait4(self):
        rerun = run_unit(self.checkpoint, self.dataset, 100)
        self.assertEqual(rerun['Now']['history'], self.results['Now']['history'])
        self.assertEqual(rerun['Now']['orders'][4], self.results['Wait4']['orders'][4])
        self.assertNotEqual(rerun['Now']['orders'][3], rerun['Now']['orders'][4])

    def test_source_not_mutated(self):
        self.assertTrue(states_equal(self.checkpoint, self.original))

    def test_inherited_momentum(self):
        for arm in ARMS:
            self.assertTrue(states_equal(self.results[arm]['before']['optimizer'],
                                        self.checkpoint['optimizer_state_dict']))
            self.assertTrue(states_equal(self.results[arm]['before']['optimizer']['state'],
                                        self.results[arm]['after']['optimizer']['state']))

    def test_now_reproduction(self):
        self.assertTrue(compare_reproduction(self.record('Now'), self.historical))

    def test_wait3_reproduction(self):
        self.assertTrue(compare_reproduction(self.record('Wait3'), self.historical))

    def test_valid_roster(self):
        self.assertEqual(validate_records(*self.roster())['status'], 'PASS')

    def test_reproduction_fields_fail_closed(self):
        for field in ('loss', 'order', 'tensor', 'dtype', 'inventory', 'epoch'):
            with self.subTest(field=field):
                records, bindings, history = self.roster()
                row = records[0]
                if field == 'loss': row['history'][0]['val_loss'] += 1e-12
                elif field == 'order': row['orders'][0][0].reverse(); row['order_sha256'] = order_hashes(row['orders'])
                elif field == 'tensor': row['evidence'][0]['model_parameters'][0]['state_npy_base64'] = 'invalid'
                elif field == 'dtype': row['evidence'][0]['model_parameters'][0]['dtype'] = 'float64'
                elif field == 'inventory': row['evidence'][0]['optimizer_state'].pop()
                else: row['evidence'][0]['epoch'] = 16
                with patch('reflexml.wait_d.summarize', side_effect=AssertionError('Must not analyze')):
                    self.assertEqual(analyze_records(records, bindings, history)['classification'], 'INVALID')

    def test_order_mismatch_invalid(self):
        records, bindings, history = self.roster()
        records[1]['orders'][0][0].reverse()
        records[1]['order_sha256'] = order_hashes(records[1]['orders'])
        self.assertEqual(validate_records(records, bindings, history)['classification'], 'INVALID')

    def test_coverage_and_duplicates_invalid(self):
        records, bindings, history = self.roster()
        for altered in (records[:-1], records + [records[0]], records[:-1]+[records[0]]):
            self.assertEqual(validate_records(altered, bindings, history)['classification'], 'INVALID')

    def test_identity_schedule_nonfinite_failures(self):
        for kind in ('binding', 'schedule', 'nonfinite', 'status', 'h5_population'):
            records, bindings, history = self.roster()
            row = records[0]
            if kind == 'binding': row['binding']['population_size'] = 13
            elif kind == 'schedule': row['history'][0]['learning_rate'] = .1
            elif kind == 'nonfinite': row['history'][0]['val_loss'] = float('nan')
            elif kind == 'status': row['status'] = 'FAILURE'
            else:
                row['orders'][4] = []; row['order_sha256'] = order_hashes(row['orders'])
            self.assertEqual(validate_records(records, bindings, history)['classification'], 'INVALID')

    def test_complete_source_restore_no_updates(self):
        data = make_loaders_from_datasets(self.dataset, [], self.config)
        with patch('reflexml.wait_d._configuration', return_value=self.config):
            validate_source_state(self.checkpoint, data)
        self.assertTrue(states_equal(self.checkpoint, self.original))

    def test_source_missing_momentum_rejected(self):
        checkpoint = deepcopy(self.checkpoint)
        checkpoint['optimizer_state_dict']['state'].popitem()
        data = make_loaders_from_datasets(self.dataset, [], self.config)
        with patch('reflexml.wait_d._configuration', return_value=self.config):
            with self.assertRaises(ValueError): validate_source_state(checkpoint, data)

    def test_valid_analysis_extracts_all_horizon_deltas(self):
        records, bindings, history = self.roster()
        result = analyze_records(records, bindings, history)
        expected = [w['val_loss'] - n['val_loss'] for w, n in
                    zip(self.results['Wait3']['history'], self.results['Now']['history'])]
        self.assertEqual(result['Delta']['001-A1-Wait3'], expected)
        self.assertAlmostEqual(result['delays'][3]['P'], expected[2])
        self.assertAlmostEqual(result['delays'][3]['Q'], expected[3])

    def test_partial_disk_run_invalid_without_analysis(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); write_json(root/'manifest.json', {})
            with patch('reflexml.wait_d.preflight', return_value=({}, {})), \
                 patch('reflexml.wait_d.summarize', side_effect=AssertionError('Must not analyze')):
                result = inspect_run(Path('.'), None, root, analyze=True)
            self.assertEqual(result['classification'], 'INVALID')

    def test_changed_manifest_invalid(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); write_json(root/'manifest.json', {'code': 'changed'})
            with patch('reflexml.wait_d.preflight', return_value=({}, {})):
                result = inspect_run(Path('.'), None, root)
            self.assertEqual(result['classification'], 'INVALID')

    def test_failed_arm_records_error_without_retry(self):
        calls = []
        with patch('reflexml.wait_d.run_branch', side_effect=ValueError('synthetic failure')) as kernel:
            with self.assertRaises(ValueError):
                run_unit(self.checkpoint, self.dataset, 100,
                         on_arm=lambda arm, result, error: calls.append((arm, result, str(error))))
        self.assertEqual(kernel.call_count, 1)
        self.assertEqual(calls, [('Now', None, 'synthetic failure')])

    def test_no_overwrite(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)/'record.json'; write_json(path, {})
            with self.assertRaises(FileExistsError): write_json(path, {})


if __name__ == '__main__': unittest.main()
