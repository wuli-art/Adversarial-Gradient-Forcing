"""Exercise the training rollout, reconstruction and replay consumer on CPU."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from agf.pipeline.self_gradient_forcing_training import SelfGradientForcingTrainingPipeline
from agf.trainer.distillation import FakeScoreReplayBuffer, Trainer
from agf.utils.config import PACKAGE_ROOT, load_config
from agf.utils.scheduler import FlowMatchScheduler


class TinyGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7))
        self.calls = []

    def forward(self, noisy_image_or_video, timestep, clean_x=None, kv_cache=None, **kwargs):
        self.calls.append(torch.is_grad_enabled())
        x = noisy_image_or_video
        if clean_x is None:
            context = kv_cache[0].get("history", torch.zeros_like(x[:, :1]))
            if not timestep.any():
                kv_cache[0]["history"] = context + x.sum(1, keepdim=True)
        else:
            context = clean_x.cumsum(1)
        flow = self.weight * (x + 0.2 * context)
        return flow, x - 0.1 * flow


def make_pipeline():
    generator = TinyGenerator()
    scheduler = FlowMatchScheduler(num_inference_steps=1000, shift=5)
    pipeline = SelfGradientForcingTrainingPipeline(
        torch.tensor([999.0, 937.0, 833.0, 624.0]),
        scheduler,
        generator,
        num_max_frames=21,
        same_step_across_blocks=True,
    )
    # Keep the real serial control flow while shrinking only the cache allocation.
    pipeline.num_transformer_blocks = 1
    pipeline.frame_seq_length = 1
    pipeline.kv_cache_size = 21
    return pipeline, generator


@pytest.mark.parametrize("exit_step", [0, 1, 3])
def test_rollout_records_shared_noise_and_reconstructs_with_gradients(exit_step):
    pipeline, generator = make_pipeline()
    pipeline._sample_exit_step = lambda **kwargs: exit_step
    noise = torch.linspace(-1, 1, 84).reshape(1, 21, 1, 2, 2)
    state = pipeline.rollout(noise)
    assert torch.equal(state.initial_noise, noise)
    assert not state.context_latents.requires_grad
    assert not state.noisy_at_t.requires_grad
    assert generator.calls == [False] * 35  # 7 chunks × (4 denoising + 1 cache write)
    assert pipeline.kv_cache is None and pipeline.crossattn_cache is None
    if exit_step == 0:
        assert torch.equal(state.noisy_at_t, noise)
    output = pipeline.reconstruct(state, {})
    output.square().mean().backward()
    assert generator.calls[-1] is True
    assert torch.isfinite(generator.weight.grad) and generator.weight.grad.abs() > 0


def test_rollout_rejects_a_different_training_window():
    pipeline, _ = make_pipeline()
    with pytest.raises(ValueError, match="training window"):
        pipeline.rollout(torch.zeros(1, 24, 1, 2, 2))


@pytest.mark.parametrize("extra_metadata", [False, True])
def test_critic_consumes_replay_with_or_without_optional_metadata(extra_metadata):
    weight = torch.nn.Parameter(torch.tensor(2.0))
    replay = FakeScoreReplayBuffer(2, 0.5, 0.5, 17)
    sample = {
        "kind": "robust",
        "generated_image": torch.ones(1, 3, 1, 1, 1),
        "conditional_dict": {},
    }
    if extra_metadata:
        sample.update(
            phase=2, rollout_prefix_chunks=0, rollout_age_bucket=0, rollout_total_chunks=7
        )
    replay.push(sample)
    model = SimpleNamespace(
        eval=lambda: None, critic_loss_from_generated=lambda **kw: (weight.square(), {})
    )
    trainer = SimpleNamespace(
        model=model, fake_replay=replay, device=torch.device("cpu"), gradient_accumulation_steps=2
    )
    logs = Trainer.fwdbwd_one_step(trainer, None, train_generator=False)
    assert weight.grad.item() == 2.0
    assert logs["critic_replay_robust"].item() == 1.0
    assert logs["critic_loss"].item() == 4.0


def test_direct_agf_changes_only_regularization_and_keeps_400_steps():
    paper = OmegaConf.to_container(load_config(), resolve=True)
    direct = OmegaConf.to_container(
        load_config(PACKAGE_ROOT / "configs/train_direct.yaml"), resolve=True
    )
    assert paper["max_steps"] == direct["max_steps"] == 400
    assert paper["checkpoint_steps"] == direct["checkpoint_steps"] == [200, 400]
    assert paper.pop("regularization_enabled") is True
    assert direct.pop("regularization_enabled") is False
    assert paper == direct
