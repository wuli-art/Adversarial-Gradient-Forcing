from collections import OrderedDict

import pytest
import torch

from agf.inference import generator_weights, prompt_indices, write_video
from agf.trainer.distillation import FakeScoreReplayBuffer, validate_resume_checkpoint
from agf.utils.dataset import TextDataset
from agf.utils.distributed import clean_fsdp_state_dict


@pytest.mark.parametrize("count,world", [(20, 8), (3, 8), (1, 1), (0, 8)])
def test_inference_shards_without_dropping_or_repeating_prompts(count, world):
    indices = [i for rank in range(world) for i in prompt_indices(count, rank, world)]
    assert sorted(indices) == list(range(count))


def test_checkpoint_section_and_wrapper_names():
    value = torch.ones(1)
    state = {"generator": OrderedDict({"_fsdp_wrapped_module.layer.weight": value})}
    assert generator_weights(state, False)["layer.weight"] is value
    with pytest.raises(ValueError, match="generator_ema"):
        generator_weights(state, True)
    with pytest.raises(ValueError, match="collision"):
        clean_fsdp_state_dict({"layer.weight": value, "_fsdp_wrapped_module.layer.weight": value})


def test_replay_resume_restores_samples_and_sampling_rng():
    replay = FakeScoreReplayBuffer(2, 0.5, 0.5, 17)
    for kind in ("clean", "robust"):
        replay.push(
            {
                "kind": kind,
                "generated_image": torch.randn(1, 3, 1, 1, 1),
                "conditional_dict": {"prompt_embeds": torch.randn(1, 2)},
                "phase": 2,
                "rollout_prefix_chunks": 0,
                "rollout_age_bucket": 0,
                "rollout_total_chunks": 7,
            }
        )
    state = replay.state_dict()
    restored = FakeScoreReplayBuffer(2, 0.5, 0.5, 99)
    restored.load_state_dict(state, torch.device("cpu"))
    for _ in range(12):
        expected, actual = replay.sample(), restored.sample()
        assert expected["kind"] == actual["kind"]
        assert torch.equal(expected["generated_image"], actual["generated_image"])


def test_empty_prompt_lines_are_rejected(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("first\n\nsecond\n")
    with pytest.raises(ValueError, match="nonempty"):
        TextDataset(path)


def test_video_can_be_decoded_and_never_overwrites_output(tmp_path):
    import imageio.v2 as imageio
    import numpy as np

    destination = tmp_path / "preview.mp4"
    frames = np.zeros((5, 32, 48, 3), dtype=np.uint8)
    frames[2:] = 127
    write_video(destination, frames)
    with imageio.get_reader(destination) as reader:
        assert reader.count_frames() == 5
        assert reader.get_data(0).shape == (32, 48, 3)
        assert reader.get_meta_data()["fps"] == 16
    with pytest.raises(FileExistsError):
        write_video(destination, frames)


@pytest.mark.parametrize("step,weight,has_ema", [(1, 0.99, False), (200, 0.99, True), (200, 0.0, False)])
def test_resume_obeys_ema_start_and_disabled_schedule(step, weight, has_ema):
    from types import SimpleNamespace

    config = SimpleNamespace(resume_step=step, ema_weight=weight, ema_start_step=200)
    checkpoint = {"generator": {}, "critic": {}, "training_step": step}
    if has_ema:
        checkpoint["generator_ema"] = {}
    assert validate_resume_checkpoint(checkpoint, config) == step
    if has_ema:
        del checkpoint["generator_ema"]
        with pytest.raises(KeyError, match="generator_ema"):
            validate_resume_checkpoint(checkpoint, config)
    else:
        checkpoint["generator_ema"] = {}
        with pytest.raises(ValueError, match="EMA"):
            validate_resume_checkpoint(checkpoint, config)
