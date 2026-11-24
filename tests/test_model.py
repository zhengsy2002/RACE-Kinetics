import copy
from pathlib import Path
import sys
import unittest

import pandas as pd
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from race_kinetics.config import load_config, validate, PROJECT_ROOT
from race_kinetics.experiments import arm_config
from race_kinetics.losses import AlignmentLoss, contrastive_loss, direction_loss
from race_kinetics.models.initialization import initialize_model
from race_kinetics.models.alignment import checkpoint_components
from race_kinetics.models.projections import checkpoint_model_options
from race_kinetics.training.supervision import DirectionMask


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_contrastive_loss_matches_probability_definition(self):
        torch.manual_seed(42)
        left, right = torch.randn(4, 5, dtype=torch.float64), torch.randn(4, 5, dtype=torch.float64)
        mask = torch.tensor([[1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 1]], dtype=torch.bool)
        logits = 0.5 * left @ right.t()
        expected = -0.5 * ((logits.softmax(1) * mask).sum(1).log().mean()
                           + (logits.softmax(0) * mask).sum(0).log().mean())
        actual, _ = contrastive_loss(left, right, 0.5, mask)
        torch.testing.assert_close(actual, expected)

    def test_direction_masked_loss_and_empty_batch(self):
        forward = torch.tensor([1.5, -0.3, 0.8], requires_grad=True)
        reverse = torch.tensor([-0.1, 0.7, -0.9], requires_grad=True)
        mask = torch.tensor([True, False, True])
        loss, _ = direction_loss(forward, reverse, mask)
        expected = (-F.logsigmoid(forward[mask]).sum() - F.logsigmoid(-reverse[mask]).sum()) / 4
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(forward.grad[1].item(), 0)
        self.assertEqual(reverse.grad[1].item(), 0)
        zero, _ = direction_loss(forward, reverse, torch.zeros(3, dtype=torch.bool))
        self.assertEqual(zero.item(), 0)
        self.assertTrue(zero.requires_grad)

    def test_conflicts_are_filtered_within_each_split(self):
        reactions = pd.DataFrame({'reaction_row': range(6),
            'substrate': ['C', 'CO', 'CC', 'CCC', 'O', 'N'],
            'product': ['CO', 'C', 'CC', 'CCCO', 'N', 'O']})
        pairs = pd.DataFrame({'reaction_row': range(6), 'split': ['train'] * 5 + ['valid']})
        masks = DirectionMask(reactions, pairs)
        self.assertEqual(masks.eligible['train'].tolist(), [False, False, False, True, True, False])
        self.assertTrue(masks.eligible['valid'][5])
        with self.assertRaisesRegex(ValueError, 'outside'):
            masks.attach({'reaction_row': torch.tensor([5])}, training=True)

    def test_loss_ablations_disable_only_the_selected_gradient_path(self):
        config = load_config(PROJECT_ROOT / 'configs/default.json')
        for name, inactive in [('full', None), ('no_er', 'protein_projection'),
                               ('no_rp', 'reactant_projection'), ('no_dir', 'direction_head')]:
            with self.subTest(name=name):
                current = arm_config(config, name)
                model = initialize_model(seed=42, dropout=0)
                batch = {'protein': torch.randn(4, 1152), 'reactant_molformer': torch.randn(4, 768),
                         'product_molformer': torch.randn(4, 768), 'protein_row': torch.tensor([0, 1, 2, 3]),
                         'reaction_row': torch.tensor([0, 0, 1, 2]),
                         'direction_eligible': torch.tensor([True, True, False, True])}
                loss, _ = AlignmentLoss(**current['loss'])(model(batch), batch)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for group in ['protein_projection', 'reactant_projection', 'direction_head']:
                    gradients = [p.grad for p in getattr(model, group).parameters()]
                    if group == inactive:
                        self.assertTrue(all(g is None for g in gradients))
                    else:
                        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in gradients))
                self.assertFalse(any('conditional' in key for key in model.state_dict()))

    def test_current_checkpoints_require_architecture_metadata(self):
        model = initialize_model(seed=42)
        state = checkpoint_components(model)
        state['molformer_dim'] = 768
        self.assertEqual(checkpoint_model_options(state)['protein_projection'], 'residual')
        del state['architecture']
        with self.assertRaisesRegex(ValueError, 'explicit current architecture'):
            checkpoint_model_options(state)

    def test_obsolete_options_are_rejected(self):
        base = load_config(PROJECT_ROOT / 'configs/default.json')
        for key in ('study', 'pretraining', 'loss_weighting'):
            with self.subTest(key=key):
                changed = copy.deepcopy(base)
                changed[key] = {}
                with self.assertRaises(ValueError):
                    validate(changed)
        changed = copy.deepcopy(base)
        changed['loss']['conditional_product'] = 0
        with self.assertRaises(ValueError):
            validate(changed)


if __name__ == '__main__':
    unittest.main()
