# (a) Consider the following model:
# 1 class ToyModel(nn.Module):
# 2 def __init__(self, in_features: int, out_features: int):
# 3 super().__init__()
# 4 self.fc1 = nn.Linear(in_features, 10, bias=False)
# 5 self.ln = nn.LayerNorm(10)
# 6 self.fc2 = nn.Linear(10, out_features, bias=False)
# 7 self.relu = nn.ReLU()
# 8
# 9 def forward(self, x):
# 10 x = self.relu(self.fc1(x))
# 11 x = self.ln(x)
# 12 x = self.fc2(x)
# 13 return x
# Suppose we are training the model on a GPU and that the model parameters are originally in
# FP32. We’d like to use autocasting mixed precision with FP16. What are the data types of:
# • the model parameters within the autocast context,
# • the output of the first feed-forward layer (ToyModel.fc1),
# • the output of layer norm (ToyModel.ln),
# • the model’s predicted logits,
# • the loss,
# • and the model’s gradients?
# Deliverable: The data types for each of the components listed above.
# (b) You should have seen that FP16 mixed precision autocasting treats the layer normalization layer
# differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed
# precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently?
# Why or why not?
# Deliverable: A 2-3 sentence response.
# (c) Modify your benchmarking script to optionally run the model using mixed precision with BF16.
# Time the forward and backward passes with and without mixed-precision for each language model
# size described in §1.1.2. Compare the results of using full vs. mixed precision, and comment on
# any trends as model size changes. You may find the nullcontext no-op context manager to be
# useful.
# Deliverable: A 2-3 sentence response with your timings and commentary.

from torch import nn
import torch
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print('1-', x.dtype)
        x = self.relu(self.fc1(x))
        print('2-', x.dtype)
        x = self.ln(x)
        print('3-', x.dtype)
        x = self.fc2(x)
        print('4-', x.dtype)
        return x

dtype = torch.float16
model = ToyModel(10, 20).to('cuda')

x = torch.randn(10, 10).to('cuda')
with torch.autocast("cuda",dtype=dtype):
    y = model(x)


# Expected output pattern:

# input: torch.float32
# after fc1+relu: torch.float16
# after ln: torch.float32
# after fc2: torch.float16
# logits: torch.float16
# loss: torch.float32
# param dtype: torch.float32
# grad dtype: torch.float32

# Big picture (one sentence)
# Autocast chooses FP16 where safe for speed, and FP32 where needed for numerical stability.