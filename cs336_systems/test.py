import os
# os.chdir('./cs336_systems')

import torch
from cs336_basics.model import BasicsTransformerLM

device = torch.device("cuda")

lm = BasicsTransformerLM(
    vocab_size=10000,
    context_length=1000,
    d_model=512,
    num_layers=4,
    num_heads=8,
    d_ff=512,
    rope_theta=0.1
).to(device)

x = torch.randint(10000, (8, 64), device=device)
y = lm(x)

print(y.shape)