import os
os.chdir("./cs336_systems/")
from cs336_basics.model import BasicsTransformerLM
import torch

vocab_size = 10000  # int,
context_length = 1000  # int,
d_model = 1024  # int,
num_layers = 24  # int,
num_heads = 8  # int,
d_ff = 1024  # int,
rope_theta = 0.1  # float,

lm = BasicsTransformerLM(vocab_size, context_length, d_model,
                         num_layers, num_heads, d_ff, rope_theta)

lm.get_num_params
lm = lm.to(device)

def fn(lm):
    x = torch.randint(vocab_size, (30, 50), device=device)
    lm = lm.to(device)
    lm.forward(x)

x = torch.randint(vocab_size, (30, 50), device=device)
x.device
print(next(lm.parameters()).device)

import timeit, time
device = torch.device('mps')
# t = timeit.timeit(fn(lm), number=1000)
# print(t / 1000)

number = 10  # warmup
for i in range(number):
    fn(lm)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    if (i + 1) % 10 == 0:
        print(f"{i+1}/{number}")

number = 1000
start = time.perf_counter()
for i in range(number):
    fn(lm)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    if (i + 1) % 10 == 0:
        print(f"{i+1}/{number}")
    print(i)
end = time.perf_counter()

print(end - start)