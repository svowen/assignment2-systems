# para
from cs336_basics.model import BasicsTransformerLM
from benchmarking_script_py import init, benchmark_lm_forward_backward

model_table = {
    "Size": ["small", "medium", "large", "xl", "2.7B"],
    "d_model": [768, 1024, 1280, 1600, 2560],
    "d_ff": [3072, 4096, 5120, 6400, 10240],
    "num_layers": [12, 24, 36, 48, 32],
    "num_heads": [12, 16, 20, 25, 32],
}

model_configs = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
    },
    "2.7B": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}

vocab_size = 50304        # GPT-2 style vocab (multiple of 128 for efficiency)
context_length = 1024     # standard sequence length
rope_theta = 10000        # standard RoPE base
batch_size = 1            # start safe (avoid OOM)

for i in model_configs:
    val = model_configs[i]
    print(i, val)
    d_model, d_ff, num_layers, num_heads = \
        val['d_model'], val['d_ff'], val['num_layers'], val['num_heads']
    lm = BasicsTransformerLM(vocab_size, context_length, d_model,
                            num_layers, num_heads, d_ff, rope_theta)



    # batch_size = 1
    # print('here to seperate forward and backward')

    ret = benchmark_lm_forward_backward(
        lm,
        vocab_size,
        batch_size,
        context_length,
        warmup_steps=5,
        measure_steps=10,
    )

    model_configs[i]['results'] = ret

    print(ret)