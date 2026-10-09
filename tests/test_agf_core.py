from pathlib import Path
import unittest

from omegaconf import OmegaConf
from agf.utils.config import TRAIN_CONFIG
import torch

from agf.model.adversarial_history_dmd import AdversarialHistoryDMD


ROOT = Path(__file__).resolve().parents[1]


class PaperAGFConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.load(TRAIN_CONFIG)

    def test_paper_agf_contract(self):
        self.assertEqual(self.config.distribution_loss, "adversarial_history_dmd")
        self.assertEqual(self.config.grad_mode, "adversarial_self_gradient_forcing")
        self.assertEqual(self.config.attack_epsilon_rms, 0.05)
        self.assertEqual(self.config.clean_loss_weight, 0.5)
        self.assertEqual(self.config.adv_loss_weight, 0.5)
        self.assertEqual(list(self.config.attack_chunk_choices), [1, 2, 3])
        self.assertTrue(self.config.attack_future_only)
        self.assertTrue(self.config.attack_temporal_shared)
        self.assertEqual(self.config.attack_steps, 1)
        self.assertFalse(self.config.attack_random_start)
        self.assertEqual(self.config.fake_replay_clean_ratio, 0.5)
        self.assertEqual(self.config.fake_replay_adv_ratio, 0.5)
        self.assertEqual(self.config.dfake_gen_update_ratio, 5)
        self.assertEqual(self.config.gc_interval, 0)
        self.assertEqual(self.config.max_steps, 400)
        self.assertTrue(self.config.regularization_enabled)

    def test_canonical_batch_and_rollout_geometry(self):
        self.assertEqual(self.config.batch_size, 1)
        self.assertEqual(self.config.total_batch_size, 64)
        self.assertEqual(self.config.num_training_frames, 21)
        self.assertEqual(self.config.num_frame_per_block, 3)
        self.assertEqual(self.config.image_or_video_shape[1], 21)


class AGFCoreMathTest(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(AdversarialHistoryDMD)
        self.model.num_frame_per_block = 3

    def test_temporal_shared_attack_changes_only_selected_chunk(self):
        context = torch.zeros(1, 21, 1, 1, 1)
        direction = torch.ones(1, 1, 1, 1, 1)
        perturbed = self.model._perturb_context(
            context, attack_chunk=1, shared_direction=direction, epsilon=0.05
        )
        expected = torch.zeros_like(context)
        expected[:, 3:6] = 0.05
        self.assertTrue(torch.equal(perturbed, expected))

    def test_future_mask_starts_after_attacked_chunk(self):
        reference = torch.zeros(1, 21, 1, 1, 1)
        mask = self.model._future_mask(reference, attack_chunk=1)
        self.assertFalse(mask[:, :6].any())
        self.assertTrue(mask[:, 6:].all())

    def test_rms_normalization_has_unit_radius(self):
        value = torch.arange(1, 13, dtype=torch.float32).reshape(1, 1, 3, 2, 2)
        normalized = self.model._normalize_rms(value)
        radius = self.model._rms(normalized)
        self.assertTrue(torch.allclose(radius, torch.ones_like(radius), atol=2e-6))

    def test_dmd_surrogate_materializes_prescribed_gradient(self):
        output = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        prescribed = torch.tensor([0.2, -0.4], dtype=torch.float64)
        loss = self.model._dmd_pseudo_loss(output, prescribed)
        loss.backward()
        self.assertTrue(torch.allclose(output.grad, prescribed / output.numel()))


if __name__ == "__main__":
    unittest.main()
