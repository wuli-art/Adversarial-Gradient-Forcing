from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


def clean_fsdp_parameter_name(name: str) -> str:
    """Remove wrapper-only path components from a parameter name."""
    return (
        name.replace("._fsdp_wrapped_module", "")
        .replace("_fsdp_wrapped_module.", "")
        .replace("._checkpoint_wrapped_module", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("._orig_mod", "")
        .replace("_orig_mod.", "")
    )


def clean_fsdp_state_dict(state_dict):
    """Return a state dict loadable by wrapped or ordinary model instances.

    Strip FSDP and activation-checkpoint wrapper components from parameter
    names so checkpoints load consistently into training and inference modules.
    """
    cleaned = state_dict.__class__()
    for name, value in state_dict.items():
        clean_name = clean_fsdp_parameter_name(name)
        if clean_name in cleaned:
            raise ValueError(
                f"Checkpoint key collision after wrapper cleanup: {name!r} "
                f"maps to existing key {clean_name!r}"
            )
        cleaned[clean_name] = value
    if hasattr(state_dict, "_metadata"):
        cleaned._metadata = {
            clean_fsdp_parameter_name(name): value
            for name, value in state_dict._metadata.items()
        }
    return cleaned


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_optim_state_dict(model, optimizer):
    """Gather a full, CPU optimizer state on rank 0 only."""
    model_config = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    optim_config = FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        model_config,
        optim_config,
    ):
        return FSDP.optim_state_dict(model, optimizer)


def fsdp_optim_state_dict_to_load(model, optimizer, full_state_dict):
    """Scatter a rank-0 full optimizer state into local FSDP shards."""
    rank0_state = (
        full_state_dict
        if not dist.is_initialized() or dist.get_rank() == 0
        else None
    )
    return FSDP.scatter_full_optim_state_dict(
        rank0_state,
        model,
        optim=optimizer,
    )


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
        timeout=timedelta(minutes=30),
        device_id=torch.device("cuda", local_rank),
    )


class EMA_FSDP:
    def __init__(
        self,
        fsdp_module: torch.nn.Module,
        decay: float = 0.999,
        rank0_only: bool = False,
    ):
        self.decay = decay
        self.rank0_only = bool(rank0_only)
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @staticmethod
    def _clean_param_name(name):
        return clean_fsdp_parameter_name(name)

    def _stores_shadow(self):
        return (
            not self.rank0_only
            or not dist.is_initialized()
            or dist.get_rank() == 0
        )

    def _summon_kwargs(self):
        return {
            "writeback": False,
            "rank0_only": self.rank0_only,
            "offload_to_cpu": self.rank0_only,
        }

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(
            fsdp_module, **self._summon_kwargs()
        ):
            if not self._stores_shadow():
                return
            for n, p in fsdp_module.module.named_parameters():
                clean_name = self._clean_param_name(n)
                self.shadow[clean_name] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(
            fsdp_module, **self._summon_kwargs()
        ):
            if not self._stores_shadow():
                return
            for n, p in fsdp_module.module.named_parameters():
                clean_name = self._clean_param_name(n)
                self.shadow[clean_name].mul_(d).add_(
                    p.detach().float().cpu(), alpha=1. - d
                )

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        return self.shadow            # picklable

    def load_state_dict(self, sd):
        if self._stores_shadow():
            self.shadow = {k: v.clone() for k, v in sd.items()}
        else:
            self.shadow = {}

    def copy_to(self, fsdp_module):
        # load EMA weights into an (unwrapped) copy of the generator
        if self.rank0_only:
            raise NotImplementedError(
                "copy_to is unavailable for rank0-only EMA; export or load "
                "the rank-0 EMA state dict instead"
            )
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in fsdp_module.module.named_parameters():
                clean_name = self._clean_param_name(n)
                if clean_name in self.shadow:
                    p.data.copy_(
                        self.shadow[clean_name].to(p.dtype, device=p.device)
                    )
