# Method and gradient boundaries

The generator follows its own chunkwise history. A serial, no-grad rollout records
initial noise, the state at a sampled denoising exit step, and the history latents.
Parallel reconstruction uses these fixed latent inputs while keeping the generator's
history K/V computation differentiable. No gradient is propagated through the
serial rollout or through selection of the adversarial direction.

## Future-guided history attack

`AdversarialHistoryDMD.generator_loss` chooses a zero-based history-block index
from `[1, 2, 3]`. A block contains three latent frames. It differentiates a
future-only DMD surrogate with respect to a shared perturbation, then takes one
RMS-normalized ascent step. Only the selected block is changed. The attack starts
at zero, uses radius 0.05, and does not create a second-order graph.

DMD computes its detached correction from the fake-score and frozen real-score
predictions on Gaussian-noised generated videos. A surrogate
`0.5 * mean((x - stopgrad(x - g))²)` injects the correction into the generator.
The clean branch averages over all latent coordinates; the adversarial branch
averages only over coordinates after the attacked block. Both weights are 0.5.
The attack multiplies the future-mean surrogate by its selected coordinate count
before RMS normalization, avoiding domination by the normalization stabilizer.

The clean branch reuses the correction sampled during attack construction. The
adversarial branch samples a new DMD timestep/noise pair. Generator parameters are
frozen only for the inner input-gradient call; clean and adversarial outer graphs
are recomputed separately and released after their backwards for FSDP compatibility.

## Prediction consistency regularization

The first denoising timestep is the **scheduler-mapped** first node, not a hardcoded
unwarped timestep. Clean and adversarial calls use the same recorded initial noisy
latents and current generator parameters. The pipeline returns the native flow
prediction, not a final latent converted to flow.

`L_reg = mean_future((v_adv - stopgrad(v_clean))²)`.

Only the adversarial prediction and its history K/V computation receive this
regularization gradient.

The coefficient is detached:

`lambda = clamp(rho * r_base / (r_reg + 1e-12), 0, 1000)`.

Define `r_base` as
`0.5 * RMS(grad_x L_clean) + 0.5 * RMS(grad_x L_adv)`. Each RMS uses the **entire
output tensor** as its denominator, with zero gradient outside the future mask.
It is a weighted sum of two output-gradient RMS proxies, not the RMS of their sum
and not a parameter-gradient norm. For the regularizer, `r_reg` is the full-output
RMS of `2 * (v_adv - v_clean) / N_future`, zero outside the future mask. The code
computes both identities directly without an auxiliary backward on an FSDP graph.

The target `rho` ramps linearly from zero at completed-iteration index 20 to
0.05 at index 80, then holds until the stage transition. The counter passed into
an update is the number of **completed** outer iterations: generator updates
occur at counter values 0, 5, …, 195 in stage 1 and
200, 205, …, 395 in stage 2. The latter stage disables regularization entirely.
Thus “200 + 200 steps” means 400 outer iterations, not 400 generator updates.

Both DMD graphs are released before constructing the extra clean/adversarial
first-step pair. The clean flow call is no-grad; the adversarial flow call
performs a separate backward accumulated into the same generator optimizer step.

## Replay

Each rank maintains separate clean and adversarial queues, each with capacity
8 microbatches. Samples are detached and stored on CPU in bf16. Sampling chooses
a branch with probability 0.5, then a uniformly random entry within that queue.
This is a probability mixture, not an exactly balanced count in every mini-batch.
The replay sampling RNG and queue contents are part of the resumable checkpoint.
