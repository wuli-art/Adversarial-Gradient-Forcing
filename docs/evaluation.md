# Paper evaluation

Generate from the step-400 EMA with the bundled test prompts. All phases use
832×480, 16 fps, four denoising steps, local12/sink3 with relative sink positions,
and one video per prompt.

```bash
AGF_RUN_ID=paper-eval bash scripts/infer_benchmark.sh 5s \
  runs/AGF/paper/stage2/logs/checkpoint_model_000400/model.pt --gpus 0,1,2,3,4,5,6,7
AGF_RUN_ID=paper-eval bash scripts/infer_benchmark.sh 60s \
  runs/AGF/paper/stage2/logs/checkpoint_model_000400/model.pt --gpus 0,1,2,3,4,5,6,7
AGF_RUN_ID=paper-eval bash scripts/infer_benchmark.sh 240s \
  runs/AGF/paper/stage2/logs/checkpoint_model_000400/model.pt --gpus 0,1,2,3,4,5,6,7
```

Inference filenames are `<five-digit-index>-00_ema.mp4` with matching JSON metadata.
The evaluator maps these files to VBench's prompt naming internally. Existing
videos are not overwritten by the inference worker.

To continue interrupted generation, repeat the same command with `--resume`:

```bash
AGF_RUN_ID=paper-eval bash scripts/infer_benchmark.sh 240s \
  runs/AGF/paper/stage2/logs/checkpoint_model_000400/model.pt \
  --gpus 0,1,2,3,4,5,6,7 --resume
```

The worker verifies checkpoint contents, configuration, prompts, seeds and video
hashes before skipping completed samples. Missing or damaged samples are regenerated;
completed samples are retained. GPU count may change without changing prompt seeds.
Keep `inference_run.json` and each video's JSON metadata with the outputs. Older
outputs without this manifest require a new output directory because their checkpoint
identity cannot be verified. A different checkpoint or inference setting also requires
a new output directory.

## Isolated evaluation environment

VBench uses older Transformers/tokenizers than AGF. Use a **separate Linux
Python 3.10 environment**; never install its requirements over the training venv.
These commands are run from the AGF checkout, after inference is finished.

```bash
python3.10 -m venv .venv-eval
.venv-eval/bin/python -m pip install --upgrade pip setuptools wheel
.venv-eval/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-eval/bin/python -m pip install -r evaluation/requirements.txt
.venv-eval/bin/python -m pip install Cython==3.1.3 boto3==1.40.22 \
  easydict==1.13 pycocotools==2.0.10
.venv-eval/bin/python -m pip install --no-deps dreamsim==0.2.1 pyiqa==0.1.13
.venv-eval/bin/python -m pip install --no-build-isolation --no-deps \
  'git+https://github.com/facebookresearch/detectron2.git@v0.6'

git clone https://github.com/Vchitect/VBench.git third_party/VBench
git -C third_party/VBench checkout 45e79ec14e69a2187202c675d2dbce1a71843d53
.venv-eval/bin/python evaluation/patch_vbench.py \
  --vbench-root third_party/VBench --manifest third_party/vbench_patch_manifest.json
```

The patcher configures local BERT loading and fixes tokenizer and VBench-Long
compatibility. It also keeps preprocessing directories out of
VBench-Long's video-suffix detection. The patcher checks that the source matches
the expected revision before applying changes. Detectron2 is compiled for the
local CUDA/PyTorch stack. Refer to the
[VBench installation guide](https://github.com/Vchitect/VBench) for system build
prerequisites and model licenses. `ffmpeg` must be on `PATH` for long-video splitting.

Download the evaluation model cache and BERT using the **training environment's**
Hugging Face CLI, which provides `hf`:

```bash
.venv/bin/hf download dc-ai/vbench_pretrained_models --local-dir evaluation_models/vbench
.venv/bin/hf download google-bert/bert-base-uncased --local-dir evaluation_models/bert-base-uncased
export VBENCH_BERT_BASE_UNCASED_DIR="$PWD/evaluation_models/bert-base-uncased"
```

The VBench cache must include AMT, CLIP, DINO, pyiqa, RAFT, torch-hub and aesthetic
assets; the full 16-dimensional protocol also needs its semantic-model assets.
In particular, retain
`aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth`. The evaluator uses a local
cache and offline Hugging Face settings.

## Run VBench

```bash
.venv-eval/bin/python evaluation/vbench.py \
  --python .venv-eval/bin/python --vbench-root third_party/VBench \
  --model-root evaluation_models/vbench --phase 240s \
  --videos runs/AGF/paper-eval/videos/240s \
  --prompts prompts/benchmark/mgvb_extended_240s_diverse_120.txt \
  --output-root runs/AGF/paper-eval/metrics/240s \
  --work-root runs/AGF/paper-eval/eval-work/240s --gpus 0,1,2,3,4,5,6,7
```

For `5s` and `60s`, change the phase, video/output/work directories and prompt
file to the corresponding row in [the prompt manifest](../prompts/benchmark/manifest.json).
The driver accepts a smaller GPU list for evaluation. Put `eval-work` on fast
local storage; long-video splitting produces many clips.

Before scoring, the driver verifies the required inputs, video count, filenames,
dimensions, frame count and FPS. Cached results are matched to the input videos
and evaluator source hashes.
Repeating a command resumes only structurally valid results for the same inputs;
changed inputs require new output/work directories. Raw results, logs, a summary
and the success marker are saved after result counts and scores are validated.

## Scores and aggregation

- **5 s:** 944 unique prompts, 81 frames, VBench's 16 dimensions. Preserve the
  official metadata's dimension memberships, including duplicate prompt records.
- **60 s:** 240 prompts, 957 frames. VBench-Long uses its mixed 10 s / 2 s clip
  policy and all 16 dimensions. Temporal flickering has a content-dependent
  static-video filter; color counts also depend on object detection.
- **240 s:** 120 prompts, 3837 frames. Use subject/background consistency,
  motion smoothness, dynamic degree, aesthetic quality and imaging quality.
  This produces 120 two-second clip records per video,
  or 14,400 records per dimension. The driver evaluates deterministic shards,
  checks every shard, and merges detail rows before computing the aggregate.

For 5/60 s, the driver applies the VBench normalized quality/semantic
formulas. For 240 s, Q5 is the arithmetic mean of the five dimensions excluding
dynamic degree; Q6 also includes dynamic degree. These are **project summaries,
not official VBench overall scores**. Dynamic degree measures motion incidence
and is not by itself evidence of coherent motion or quality.
