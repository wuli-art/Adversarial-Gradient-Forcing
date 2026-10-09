# Training and continuation

The paper checkpoint uses regularized AGF through global iteration 200, then pure
AGF through iteration 400, with **no reset** of generator, critic, EMA, optimizers,
replay, RNG or data-loader position. The regularizer is the only objective removed.

## Paper recipe

```bash
AGF_RUN_ID=paper bash scripts/train_paper.sh --dry-run
AGF_RUN_ID=paper bash scripts/train_paper.sh --gpus 0,1,2,3,4,5,6,7
```

| Setting | Value |
| --- | --- |
| Generator / fake score / frozen teacher | Wan 1.3B / 1.3B / 14B |
| Training window | 21 latent frames, 3 per chunk, 480×832 |
| History attack | One future-guided RMS-normalized step, radius 0.05 |
| Clean / adversarial loss and replay mixture | 0.5 / 0.5 |
| Effective batch | 64: 1 per GPU × 8 GPUs × 8 accumulation steps |
| Generator / fake-score learning rates | 2e-6 / 4e-7 |
| Iterations 1–200 | AGF + first-step prediction consistency |
| Iterations 201–400 | Pure AGF, full state retained at the transition |
| Total optimizer updates | 80 generator, 400 fake score |
| Evaluation checkpoint | Step-400 generator EMA, decay 0.99 |

## Individual stages

```bash
AGF_RUN_ID=paper-stage1 bash scripts/train.sh \
  --config configs/train_stage1.yaml --gpus 0,1,2,3,4,5,6,7

AGF_RUN_ID=paper-stage2 bash scripts/train.sh \
  --config configs/train_stage2.yaml --gpus 0,1,2,3,4,5,6,7 \
  --resume runs/AGF/paper-stage1/logs/checkpoint_model_000200/model.pt \
  --resume-step 200
```

These are the same two model launches as `scripts/train_paper.sh`. The wrapper
adds a structural checkpoint verification and a stage-transition record. It uses
separate `stage1` and `stage2` directories and fails rather than overwriting a run.
To launch only stage 2 through the wrapper:

```bash
AGF_RUN_ID=paper-continuation bash scripts/train_paper.sh \
  --resume-stage2 runs/AGF/paper-stage1/logs/checkpoint_model_000200/model.pt
```

`configs/train.yaml` is also a complete 400-iteration configuration: its schedule
turns the regularizer off at completed step 200 even within one process. The
two-stage launcher saves and reloads the complete training state at step 200.

## Interrupted runs

Keep the **whole checkpoint directory**. Resume with the configuration for the
stage that was interrupted, the saved global step, and the same world size.
Choose a fresh run ID or log directory for the continuation; the source checkpoint
and previous outputs remain in place:

```bash
AGF_RUN_ID=paper-recovered bash scripts/train.sh \
  --config configs/train_stage2.yaml \
  --resume runs/AGF/paper/stage2/logs/checkpoint_model_000300/model.pt \
  --resume-step 300
```

`max_steps=400` includes completed iterations. The next update after loading 300
is iteration 301. Changing `resume_reset_optimizers` or `resume_reset_replay`
creates a reset experiment and is not exact state continuation.

Checkpoints use **format v3**:

```text
checkpoint_model_000200/
  model.pt                       generator, critic, EMA and step metadata
  trainer_state/
    optimizers.pt                both optimizer states and global step
    rank_00000.pt … rank_00007.pt replay, RNG and data-loader state for each rank
```

The loader also supports format-v2 checkpoints with embedded optimizer state.
Files are written via `.partial` files and atomic rename. The model file is
published after all rank and optimizer files; the wrapper validates every rank
before resuming. Copy the complete checkpoint directory to local storage before launch.

## Recipe settings and ablations

YAML overrides packaged defaults; repeated `--set KEY=VALUE` overrides YAML.
Relative input paths are resolved against the invocation directory. Unknown keys
and unsupported method variants fail before model allocation.

```bash
# Short launch check of the regularized path: activate the ratio immediately.
AGF_RUN_ID=smoke bash scripts/train.sh --set max_steps=6 \
  --set total_batch_size=8 --set regularization_ramp_start=0 \
  --set regularization_ramp_end=1 --no-save

# Pure AGF from ODE initialization, without prediction consistency.
AGF_RUN_ID=direct bash scripts/train.sh --config configs/train_direct.yaml
```

The direct-AGF configuration uses the same 400 iterations, optimizer settings and
checkpoint schedule as the main configuration, with regularization disabled.
Attack radius, branch weights (which must sum to one), learning rates and the
regularizer's target ratio are configurable.

Training uses AdamW (`beta=(0,0.999)`, generator weight decay 0.01), bf16,
activation checkpointing, FSDP `hybrid_full`, gradient norm clipping at 10,
seed 20260731, teacher guidance 3.0 and four shifted denoising nodes
`[1000,750,500,250]`. EMA starts at global step 200 with decay 0.99.

The default configuration uses eight GPUs and gradient accumulation to reach a
global batch size of 64. Resume with the same GPU count as the saved checkpoint.
Each rank drops an incomplete final microbatch to maintain the configured effective
batch size. Legacy checkpoints remain compatible when their loader produced full
batches, including the default batch size of one. A legacy checkpoint that consumed
partial batches must be continued with its original code or used for a new experiment.

Optional W&B logging is enabled by `pip install -e '.[logging]'` and `--wandb`.
Credentials use W&B's normal login/environment mechanism. Resolved configurations
and environment versions are saved with the run. Regularization logs include its
raw loss, coefficient, target ratio and achieved output-gradient ratio.

## Output paths

`AGF_RUNS_ROOT` sets the run root (default: `./runs`) and `AGF_RUN_ID` names the
experiment. Default outputs go to `<root>/AGF/<run-id>/`. The launcher passes the
same root and run ID to both stages and every distributed worker.

Relative `--logdir` and `--output` paths are resolved inside the run directory.
Absolute paths are used directly, including paths outside the run root. For example:

```bash
AGF_RUNS_ROOT=/mnt/experiments AGF_RUN_ID=paper bash scripts/train_paper.sh
bash scripts/infer.sh --checkpoint /mnt/checkpoints/model.pt --output /mnt/videos/agf
```

Training requires a new log directory, including for checkpoint continuation.
Each stage saves its own `resolved_config.yaml` and `environment.json` alongside
its logs. Existing checkpoint directories are never replaced.
