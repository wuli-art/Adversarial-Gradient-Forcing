from types import SimpleNamespace
import torch
import pytest
from agf.model.prediction_consistency import (
    target_ratio,
    calibrate_scale,
    dmd_output_gradient_rms,
    masked_mse_output_gradient_rms,
    backward_prediction_consistency,
)
from agf.model.adversarial_history_dmd import AdversarialHistoryDMD
from agf.utils.config import load_config
from agf.pipeline.self_gradient_forcing_training import SelfGradientForcingTrainingPipeline


def test_schedule_and_detached_scale():
    c = load_config()
    for step, ratio in [(0, 0), (20, 0), (50, 0.025), (80, 0.05), (195, 0.05), (200, 0), (399, 0)]:
        assert target_ratio(c, step) == pytest.approx(ratio)
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(0.4, requires_grad=True)
    scale = calibrate_scale(a, b, 0.05)
    assert scale.item() == pytest.approx(0.25)
    assert not scale.requires_grad
    assert calibrate_scale(a, torch.tensor(0.0), 0.05).item() == 1000


def test_output_proxies_match_autograd_including_masked_zeros():
    x = torch.linspace(-2, 3, 24).reshape(1, 6, 1, 2, 2).requires_grad_()
    g = torch.linspace(0.1, 0.9, 24).reshape_as(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, 3:] = True
    for m in (None, mask):
        loss = AdversarialHistoryDMD._dmd_pseudo_loss(x, g, m)
        actual = torch.autograd.grad(loss, x)[0].square().mean().sqrt()
        assert torch.allclose(dmd_output_gradient_rms(g, x, m), actual)
    y = (x.detach() * 0.3).requires_grad_()
    loss = (x[mask] - y.detach()[mask]).square().mean()
    actual = torch.autograd.grad(loss, x)[0].square().mean().sqrt()
    assert torch.allclose(masked_mse_output_gradient_rms(x, y, mask), actual)


def test_first_step_uses_shared_initial_noise_and_native_flow():
    calls = []

    def generator(**kw):
        calls.append(kw)
        return kw["clean_x"] * 3, kw["clean_x"] * 99

    pipeline = SimpleNamespace(
        generator=generator,
        denoising_step_list=torch.tensor([999.0, 937.0]),
        self_gradient_forcing_match_context=False,
    )
    noise = torch.randn(1, 6, 1, 1, 1)
    state = SimpleNamespace(
        initial_noise=noise,
        context_latents=torch.ones_like(noise),
        reconstruction_timestep=torch.zeros(1, 6),
    )
    flow = SelfGradientForcingTrainingPipeline.predict_first_step_flow_velocity(pipeline, state, {})
    assert torch.equal(flow, state.context_latents * 3)
    assert calls[0]["noisy_image_or_video"] is noise
    assert (calls[0]["timestep"] == 999).all()


def test_clean_reference_detached_and_only_future_drives_regularizer():
    weight = torch.nn.Parameter(torch.tensor(1.0))
    clean_context = torch.ones(1, 6, 1, 1, 1, requires_grad=True)
    adv_context = clean_context.detach().clone()
    adv_context[:, 1:3] += 0.05
    calls = []

    def flow(*, state, conditional_dict, context_latents):
        calls.append(torch.is_grad_enabled())
        return weight * context_latents.cumsum(1)

    model = SimpleNamespace(
        args=load_config(),
        inference_pipeline=SimpleNamespace(predict_first_step_flow_velocity=flow),
    )
    mask = torch.zeros_like(clean_context, dtype=torch.bool)
    mask[:, 3:] = True
    result = backward_prediction_consistency(
        model,
        state=SimpleNamespace(context_latents=clean_context),
        conditional_dict={},
        adversarial_context=adv_context,
        future_mask=mask,
        base_rms=torch.tensor(0.01),
        ratio=0.05,
        loss_scale=1.0,
    )
    assert calls == [False, True]
    assert clean_context.grad is None
    assert weight.grad > 0
    assert result["regularization_achieved_ratio"] == pytest.approx(0.05)


def test_real_generator_loss_runs_all_branches_then_regularizer():
    model = object.__new__(AdversarialHistoryDMD)
    torch.nn.Module.__init__(model)
    model.args = load_config()
    model.device = torch.device("cpu")
    model.generator = torch.nn.Linear(1, 1, bias=False)
    model.generator.weight.data.fill_(1.0)
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
        calls.append(("reconstruct", torch.is_grad_enabled(), model.generator.weight.requires_grad))
        return model.generator((history if context_latents is None else context_latents).cumsum(1))

    def flow(state, conditional_dict, context_latents):
        calls.append(("flow", torch.is_grad_enabled(), model.generator.weight.requires_grad))
        return model.generator(context_latents.cumsum(1))

    model.inference_pipeline = SimpleNamespace(
        reconstruct=reconstruct, predict_first_step_flow_velocity=flow
    )
    model._rollout = lambda *a, **kw: state
    model._sample_attack_chunk = lambda: 1
    model._distribution_matching_gradient = lambda image_or_video, **kw: (
        torch.ones_like(image_or_video),
        {"dmdtrain_gradient_norm": torch.tensor(1.0)},
    )
    loss, log = model.generator_loss(list(history.shape), {}, {}, None, training_step=80)
    assert len(calls) == 5
    assert calls[-2:] == [("flow", False, True), ("flow", True, True)]
    assert log["regularization_loss"] > 0 and not loss.requires_grad
    assert torch.isfinite(model.generator.weight.grad).all()
    assert len(model.pop_replay_samples()) == 2
    calls.clear()
    model.generator.zero_grad()
    _, log = model.generator_loss(list(history.shape), {}, {}, None, training_step=200)
    assert len(calls) == 3 and "regularization_loss" not in log
