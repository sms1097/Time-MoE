import os
import random

import psutil
import torch

from time_moe.datasets.time_moe_dataset import TimeMoEDataset
from time_moe.models.configuration_time_moe import TimeMoeConfig
from time_moe.models.modeling_time_moe import TimeMoeForPrediction
from time_moe.runner import TimeMoeRunner

tiny_config = TimeMoeConfig(
    input_size=1,
    hidden_size=64,  # down from 384
    intermediate_size=128,  # ~2x hidden (small MLP)
    horizon_lengths=[1, 8],
    num_hidden_layers=2,  # down from 12
    num_attention_heads=4,  # must divide hidden_size
    num_key_value_heads=4,
    hidden_act="silu",
    # Disable MoE
    num_experts_per_tok=2,
    num_experts=8,
    use_dense=False,
    apply_aux_loss=True,
    max_position_embeddings=32768,
    rms_norm_eps=1e-5,
    use_cache=False,
    rope_theta=10000,
    attention_dropout=0.0,
    # keep patching
    patch=False,
    patch_len=16,
    patch_stride=4,
)


big_config = TimeMoeConfig(
    input_size=1,
    hidden_size=384,
    intermediate_size=1536,
    horizon_lengths=1,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=None,
    hidden_act="silu",
    num_experts_per_tok=2,
    num_experts=8,
    max_position_embeddings=32768,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    use_dense=False,
    rope_theta=10000,
    attention_dropout=0.0,
    apply_aux_loss=True,
    router_aux_loss_factor=0.02,
    tie_word_embeddings=False,
    patch=False,
    patch_len=32,
    patch_stride=32,
)

model = TimeMoeForPrediction(big_config)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")


ds = TimeMoEDataset("Time-300B")
seq_idx = random.randint(0, len(ds) - 1)
seq = torch.from_numpy(ds[seq_idx]).unsqueeze(0)

smol_seq = seq[:, :4096]


# TOOD: replace with GPU version
process = psutil.Process(os.getpid())
before = process.memory_info().rss
_ = model(smol_seq)
after = process.memory_info().rss

print("Success!!")
print("Approximate activation memory", (after - before) / 1024**2, "MB")
