"""First-step prediction consistency for the 200 + 200 AGF training schedule.

Calibration uses full-output RMS proxies (including zero entries outside the
future mask), not generator parameter gradients. No auxiliary backward traverses
an FSDP graph. The clean target and the calibrated coefficient are detached.
"""

from typing import Optional
import torch
import torch.nn.functional as F


def dmd_output_gradient_rms(
    gradient: torch.Tensor,
    latent: torch.Tensor,
    gradient_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Measure the DMD surrogate's exact latent-gradient RMS.

    For ``0.5 * mean((x - (x-g).detach())**2)``, the derivative of each
    selected latent is ``g / N``.  Computing that identity directly avoids
    an auxiliary backward through a live FSDP output.
    """
    if gradient.shape != latent.shape:
        raise ValueError("DMD gradient and latent shapes must match")
    if gradient_mask is None:
        selected_count = latent.numel()
        squared_sum = gradient.float().square().sum()
    else:
        if gradient_mask.shape != latent.shape:
            raise ValueError("DMD gradient mask and latent shapes must match")
        selected_count = int(gradient_mask.sum().item())
        if selected_count <= 0:
            raise ValueError("DMD gradient mask must select at least one value")
        squared_sum = gradient.float()[gradient_mask].square().sum()
    return torch.sqrt(
        squared_sum / float(selected_count * selected_count) / float(latent.numel())
    ).detach()


def masked_mse_output_gradient_rms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Exact full-output RMS gradient of a future-masked mean MSE."""
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("masked MSE prediction, target, and mask must match")
    if mask.dtype is not torch.bool:
        raise ValueError("masked MSE mask must be boolean")
    selected_count = int(mask.sum().item())
    if selected_count <= 0:
        raise ValueError("masked MSE mask must select at least one value")
    selected_difference = prediction.detach().float()[mask] - target.detach().float()[mask]
    squared_gradient_sum = (2.0 * selected_difference / float(selected_count)).square().sum()
    return torch.sqrt(squared_gradient_sum / float(prediction.numel())).detach()


def target_ratio(config, training_step: int) -> float:
    """Schedule indexed by completed outer iterations."""
    if not config.regularization_enabled or training_step >= config.regularization_end_step:
        return 0.0
    fraction = min(
        max(
            (training_step - config.regularization_ramp_start)
            / (config.regularization_ramp_end - config.regularization_ramp_start),
            0.0,
        ),
        1.0,
    )
    return fraction * config.regularization_target_ratio


def calibrate_scale(base_rms, regularizer_rms, ratio, *, eps=1e-12, max_scale=1000.0):
    return (
        torch.nan_to_num(
            ratio * base_rms / (regularizer_rms + eps),
            nan=0.0,
            posinf=max_scale,
            neginf=0.0,
        )
        .clamp(min=0.0, max=max_scale)
        .detach()
    )


def backward_prediction_consistency(
    model, *, state, conditional_dict, adversarial_context, future_mask, base_rms, ratio, loss_scale
):
    """A fresh clean/adv flow pair after the two DMD graphs have been released."""
    with torch.no_grad():
        clean = model.inference_pipeline.predict_first_step_flow_velocity(
            state=state,
            conditional_dict=conditional_dict,
            context_latents=state.context_latents,
        ).detach()
    adv = model.inference_pipeline.predict_first_step_flow_velocity(
        state=state,
        conditional_dict=conditional_dict,
        context_latents=adversarial_context.detach(),
    )
    raw_loss = F.mse_loss(adv[future_mask].float(), clean[future_mask].float())
    raw_rms = masked_mse_output_gradient_rms(adv, clean, future_mask)
    scale = calibrate_scale(
        base_rms,
        raw_rms,
        ratio,
        eps=model.args.regularization_gradient_eps,
        max_scale=model.args.regularization_max_scale,
    )
    weighted_loss = scale * raw_loss
    (float(loss_scale) * weighted_loss).backward()
    return {
        "regularization_loss": weighted_loss.detach(),
        "regularization_raw_loss": raw_loss.detach(),
        "regularization_scale": scale,
        "regularization_base_rms": base_rms.detach(),
        "regularization_raw_rms": raw_rms,
        "regularization_target_ratio": torch.as_tensor(ratio, device=adv.device),
        "regularization_achieved_ratio": (
            scale * raw_rms / (base_rms + model.args.regularization_gradient_eps)
        ).detach(),
    }
