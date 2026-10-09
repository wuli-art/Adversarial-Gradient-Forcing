"""Exercise the actual inner attack and both outer backwards on a tiny model."""

from types import SimpleNamespace

import torch

from agf.model.adversarial_history_dmd import AdversarialHistoryDMD


def test_two_branch_backward_preserves_inner_freeze_and_replay():
    model = object.__new__(AdversarialHistoryDMD)
    torch.nn.Module.__init__(model)
    model.generator = torch.nn.Linear(1, 1, bias=False)
    model.generator.weight.data.fill_(1.0)
    model.args = SimpleNamespace(regularization_enabled=False)
    model.device = torch.device("cpu")
    model.num_frame_per_block = 3
    model.attack_epsilon_rms = 0.05
    model.clean_loss_weight = model.adv_loss_weight = 0.5
    model.fake_replay_device = torch.device("cpu")
    model.fake_replay_dtype = torch.float32
    history = torch.ones(1, 21, 1, 1, 1)
    state = SimpleNamespace(
        context_latents=history,
        denoised_timestep_from=1000,
        denoised_timestep_to=0,
        exit_step_index=0,
    )
    calls = []

    def reconstruct(state, conditional_dict, context_latents=None):
        context = state.context_latents if context_latents is None else context_latents
        calls.append((model.generator.weight.requires_grad, context.detach().clone()))
        # The future depends on prior history, so a positive surrogate gradient
        # should perturb the selected history in the positive direction.
        return model.generator(context.cumsum(dim=1))

    model.inference_pipeline = SimpleNamespace(reconstruct=reconstruct)
    model._rollout = lambda *a, **kw: state
    model._sample_attack_chunk = lambda: 1
    model._distribution_matching_gradient = lambda image_or_video, **kw: (
        torch.ones_like(image_or_video),
        {"dmdtrain_gradient_norm": torch.tensor(1.0)},
    )
    loss, logs = model.generator_loss(list(history.shape), {}, {}, None)
    assert [enabled for enabled, _ in calls] == [False, True, True]
    assert model.generator.weight.requires_grad
    assert torch.isfinite(model.generator.weight.grad).all()
    assert model.generator.weight.grad.item() > 0
    assert not loss.requires_grad
    assert torch.allclose(calls[2][1][:, 3:6], history[:, 3:6] + 0.05, atol=1e-5)
    assert torch.equal(calls[2][1][:, :3], history[:, :3])
    assert torch.equal(calls[2][1][:, 6:], history[:, 6:])
    assert abs(logs["attack_delta_rms"].item() - 0.05) < 1e-5
    replay = model.pop_replay_samples()
    assert [item["kind"] for item in replay] == ["clean", "robust"]
    assert all(not item["generated_image"].requires_grad for item in replay)
    assert model.pop_replay_samples() == []
