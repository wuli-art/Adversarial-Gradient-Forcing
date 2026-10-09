import json

from agf.utils.assets import check_model, training_asset_errors
from agf.utils.config import load_config


def test_asset_verifier_checks_all_indexed_shards(tmp_path):
    root = tmp_path / "Wan2.1-T2V-14B"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.a": "one.safetensors", "layer.b": "two.safetensors"}})
    )
    (root / "one.safetensors").write_bytes(b"test")
    errors = check_model(tmp_path, root.name)
    assert len(errors) == 1 and "two.safetensors" in errors[0]
    (root / "two.safetensors").write_bytes(b"test")
    assert check_model(tmp_path, root.name) == []


def test_resume_asset_verifier_requires_every_rank(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("one\ntwo\n")
    config = load_config(
        overrides=[f"resume_checkpoint={checkpoint}", "resume_step=200", f"data_path={prompts}"]
    )
    errors = training_asset_errors(config, tmp_path, 2)
    assert any("rank_00000.pt" in error for error in errors)
    assert any("rank_00001.pt" in error for error in errors)
