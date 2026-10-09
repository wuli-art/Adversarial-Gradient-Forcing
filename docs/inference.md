# Setup and inference


Use Linux, Python 3.10–3.12 and NVIDIA CUDA GPUs. Install PyTorch for your driver
first. The commands below use PyTorch 2.8 / CUDA 12.8.
Run them from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip install flash-attn==2.8.0.post2 --no-build-isolation
agf verify --runtime
```

Dependencies are listed in [pyproject.toml](../pyproject.toml).
`agf verify --runtime` checks imports and reports CUDA and FlashAttention availability.

### Models and training prompts

```bash
bash scripts/download_weights.sh
agf verify --require-assets
```

| Input | Default location | Used by |
| --- | --- | --- |
| Wan 1.3B | `wan_models/Wan2.1-T2V-1.3B/` | Generator, fake score, text encoder and VAE |
| Wan 14B | `wan_models/Wan2.1-T2V-14B/` | Frozen training teacher |
| Self-Forcing ODE initialization | `checkpoints/ode_init.pt` | Generator initialization |
| Self-Forcing extended VidProM prompts | `prompts/vidprom_filtered_extended.txt` | Training text |

The script downloads all four inputs above. Inference requires Wan 1.3B and a
generator checkpoint; [training](training.md) produces the AGF checkpoint.
To reuse existing inputs, pass `--wan-models`, `--ode-checkpoint`, and
`--training-prompts` to the launch commands. The
[test prompts](../prompts/README.md) are included in the repository.

## Generate videos


Generate with the trained paper checkpoint:

```bash
AGF_RUN_ID=preview bash scripts/infer.sh \
  --checkpoint runs/AGF/paper/stage2/logs/checkpoint_model_000400/model.pt \
  --prompts prompts/examples.txt --latent-frames 21 --gpus 0
```

To try inference with the ODE initialization before training:

```bash
AGF_RUN_ID=ode-preview bash scripts/infer.sh \
  --checkpoint checkpoints/ode_init.pt --no-ema --gpus 0
```

Outputs are MP4 files with per-video JSON prompt/seed metadata. EMA is selected
by default; use `--no-ema` for the ordinary generator. The default policy is
**832×480, 16 fps, local attention 12, sink 3, relative sink positions enabled**.
To continue interrupted inference, repeat the command with `--resume` and the
same run ID or output directory. Completed videos are verified before being skipped.

| Nominal duration | `--latent-frames` | Decoded frames | Actual duration |
| --- | ---: | ---: | ---: |
| 5 s | 21 | 81 | 5.0625 s |
| 60 s | 240 | 957 | 59.8125 s |
| 240 s | 960 | 3837 | 239.8125 s |

Latent-frame counts must be positive multiples of three. Output buffers and
VAE decoding memory grow with duration. The no-sink configuration is
available as `--config configs/inference_no_sink.yaml` for controlled comparisons.

