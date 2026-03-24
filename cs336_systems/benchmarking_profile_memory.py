# ... # warm-up phase in your benchmarking script
# 3
# 4 # Start recording memory history.
torch.cuda.memory._record_memory_history(max_entries=1000000)
# 6
# 7 ... # what you want to profile in your benchmarking script
# 8
# 9 # Save a pickle file to be loaded by PyTorch's online tool.
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
# 11
# 12 # Stop recording history.
torch.cuda.memory._record_memory_history(enabled=None)