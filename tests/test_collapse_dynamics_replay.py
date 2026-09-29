"""Synthetic instrumentation checks only; no historical replay is run."""

from copy import deepcopy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from reflexml.branching import states_equal
from reflexml.collapse_dynamics_replay import PROBES, PROTOCOL_ID, _snapshot, summarize_a1
from reflexml.model import FashionMLP
from reflexml.phase5 import sha256_file
from reflexml.training import train_one_epoch


class InstrumentationTests(unittest.TestCase):
    def test_probe_grid_is_frozen(self):
        self.assertEqual(PROBES, (0, 1, 3, 8, 16, 32, 48, 64, 78, 79))

    def test_snapshots_leave_numerical_path_and_rng_unchanged(self):
        images = torch.arange(8 * 28 * 28, dtype=torch.float32).reshape(8, 1, 28, 28) / 10000
        labels = torch.arange(8) % 10
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            initial = FashionMLP(4)
        source = deepcopy(initial.state_dict())
        outcomes = []
        for instrumented in (False, True):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(101)
                model = FashionMLP(4)
                model.load_state_dict(deepcopy(source))
                optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
                captures = {}

                def capture(step, current):
                    captures[step] = _snapshot(current)

                loss = train_one_epoch(
                    model, loader, optimizer, nn.CrossEntropyLoss(), torch.device("cpu"),
                    after_step=capture if instrumented else None,
                )
                outcomes.append((loss, deepcopy(model.state_dict()),
                                 deepcopy(optimizer.state_dict()), torch.random.get_rng_state().clone()))
                if instrumented:
                    self.assertEqual(set(captures), {1, 2})
                    self.assertTrue(states_equal(captures[2], model.state_dict()))
                    captures[1]["network.1.weight"].zero_()
                    self.assertFalse(states_equal(captures[1], model.state_dict()))
        self.assertEqual(outcomes[0][0], outcomes[1][0])
        self.assertTrue(states_equal(outcomes[0][1], outcomes[1][1]))
        self.assertTrue(states_equal(outcomes[0][2], outcomes[1][2]))
        self.assertTrue(torch.equal(outcomes[0][3], outcomes[1][3]))

    def test_a1_summary_uses_paired_state_gaps_and_preserves_overshoot(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for state in range(1, 37):
                folder = root / f"state_{state:03d}"
                folder.mkdir()
                path = folder / "trajectory.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=(
                        "base_run_id", "replica", "arm", "probe_step",
                        "validation_loss", "gate_pass",
                    ))
                    writer.writeheader()
                    for arm in ("Wait3", "Now"):
                        for step in PROBES:
                            now = state / 100
                            gap = 0.2 if step == 0 else (-0.1 if step == 1 else 0.0)
                            writer.writerow({
                                "base_run_id": state, "replica": "A1", "arm": arm,
                                "probe_step": step,
                                "validation_loss": now + gap if arm == "Wait3" else now,
                                "gate_pass": True,
                            })
                (folder / "report_manifest.json").write_text(json.dumps({
                    "protocol_id": PROTOCOL_ID, "covered_states": [state],
                    "probe_steps": list(PROBES), "trajectory_sha256": sha256_file(path),
                }))
            result = summarize_a1(root, root / "summary")
            with result.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(PROBES))
            self.assertAlmostEqual(float(rows[0]["C_t_A1"]), 0.2)
            self.assertAlmostEqual(float(rows[0]["F_t"]), 0.0)
            self.assertGreater(float(rows[1]["F_t"]), 1.0)
            self.assertAlmostEqual(float(rows[-1]["F_t"]), 1.0)


if __name__ == "__main__":
    unittest.main()
