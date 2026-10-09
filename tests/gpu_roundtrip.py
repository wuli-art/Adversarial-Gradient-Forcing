"""Optional two-GPU check of input gradients, accumulation and FSDP resume.

Run with torchrun --standalone --nproc-per-node=2 tests/gpu_roundtrip.py.
This uses tiny tensors and does not load Wan weights.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from agf.model.adversarial_history_dmd import AdversarialHistoryDMD
from agf.utils.distributed import (
    fsdp_optim_state_dict,
    fsdp_optim_state_dict_to_load,
    fsdp_state_dict,
    fsdp_wrap,
    launch_distributed_job,
)
from agf.utils.run_paths import resolve_run_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="checkpoints/fsdp-roundtrip")
    args = parser.parse_args()
    output = resolve_run_output("AGF", args.output, "checkpoints/fsdp-roundtrip")
    launch_distributed_job()
    try:
        rank = dist.get_rank()
        device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(1234)
        model = fsdp_wrap(
            torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 1)),
            sharding_strategy="full",
            min_num_params=1,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        x = torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4) / 10

        def step():
            optimizer.zero_grad(set_to_none=True)
            for microstep in range(2):
                from contextlib import nullcontext

                with model.no_sync() if microstep == 0 else nullcontext():
                    perturbation = torch.zeros_like(x, requires_grad=True)
                    with AdversarialHistoryDMD._input_gradient_only(
                        SimpleNamespace(generator=model)
                    ):
                        inner = model(x + perturbation).square().mean()
                        direction = torch.autograd.grad(inner, perturbation)[0].detach()
                    assert torch.isfinite(direction).all()
                    assert direction.abs().sum() > 0
                    loss = model(x + 0.05 * direction).square().mean() / 2
                    loss.backward()
            model.clip_grad_norm_(10.0)
            optimizer.step()

        step()
        state = {
            "model": fsdp_state_dict(model),
            "optimizer": fsdp_optim_state_dict(model, optimizer),
        }
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            torch.save(state, output / "state.pt")
        dist.barrier()
        step()
        expected = fsdp_state_dict(model)
        checkpoint = torch.load(output / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(
            fsdp_optim_state_dict_to_load(model, optimizer, checkpoint["optimizer"])
        )
        step()
        actual = fsdp_state_dict(model)
        if rank == 0:
            max_error = max((expected[k] - actual[k]).abs().max().item() for k in expected)
            assert max_error < 1e-7, max_error
            result = {
                "world_size": dist.get_world_size(),
                "microbatches": 2,
                "max_parameter_error_after_resume": max_error,
                "ok": True,
            }
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
