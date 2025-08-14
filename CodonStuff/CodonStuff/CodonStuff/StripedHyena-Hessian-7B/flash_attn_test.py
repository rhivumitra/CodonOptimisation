import torch
from flash_attn import flash_attn_qkvpacked_func

batch_size = 2
seqlen = 128
nheads = 8
headdim = 64

qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device="cuda", dtype=torch.float16)

out = flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False
)

print("Output shape:", out.shape)
