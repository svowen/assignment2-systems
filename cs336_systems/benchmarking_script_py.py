import os
# os.chdir("./cs336_systems/")
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW as optimizer
import torch
import torch.nn.functional as F
import time
import statistics

device = "cuda" if torch.cuda.is_available() else "cpu"

def init():
    vocab_size = 10000  # int,
    context_length = 1000  # int,
    d_model = 1024  # int,
    num_layers = 24  # int,
    num_heads = 8  # int,
    d_ff = 1024  # int,
    rope_theta = 0.1  # float,

    lm = BasicsTransformerLM(vocab_size, context_length, d_model,
                            num_layers, num_heads, d_ff, rope_theta)

    device = torch.device('cuda')
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
    # t = timeit.timeit(fn(lm), number=1000)
    # print(t / 1000)

    number = 10  # warmup
    for i in range(number):
        fn(lm)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{number}")

    number = 100
    start = time.perf_counter()
    for i in range(number):
        fn(lm)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{number}")

    end = time.perf_counter()
    print(end - start)
    return lm

def benchmark_lm_forward_backward(
    lm,
    vocab_size,
    batch_size,
    context_length,
    warmup_steps=5,
    measure_steps=10,
):
    device = torch.device('cuda')

    lm = lm.to(device)
    lm.train()

    forward_times = []
    backward_times = []

    # fake token inputs and targets
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )

    # ------------------
    # Warmup
    # ------------------
    for _ in range(warmup_steps):
        lm.zero_grad(set_to_none=True)

        if device == "cuda":
            torch.cuda.synchronize()

        logits = lm(input_ids)   # expected shape: [B, T, V]
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        if device == "cuda":
            torch.cuda.synchronize()

        loss.backward()

        if device == "cuda":
            torch.cuda.synchronize()

    # ------------------
    # Measurement
    # ------------------
    for _ in range(measure_steps):
        lm.zero_grad(set_to_none=True)

        # Forward timing
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = lm(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Backward timing
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        loss.backward()

        if device == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        forward_times.append((t1 - t0) * 1000.0)   # ms
        backward_times.append((t3 - t2) * 1000.0)  # ms

    return {
        "forward_mean_ms": statistics.mean(forward_times),
        "forward_std_ms": statistics.stdev(forward_times),
        "backward_mean_ms": statistics.mean(backward_times),
        "backward_std_ms": statistics.stdev(backward_times),
        "forward_times_ms": forward_times,
        "backward_times_ms": backward_times,
    }

