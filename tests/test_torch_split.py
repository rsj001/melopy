import torch
import torch.nn as nn
import time

# 参数设置
batch_size = 128
seq_len = 512
embed_dim = 64
num_categories = 5
vocab_sizes = [128] * num_categories
vocab_size = [90, 90, 80 ,32, 66]
full_size = sum(vocab_size)
torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模拟输入
input_ids = torch.randint(0, 128, (batch_size, seq_len, full_size))
start = time.time()

# A:
for _ in range(1000):
    input_ids.split(vocab_size, dim=-1)
time_1 = time.time() - start

start = time.time()
# B:
for _ in range(1000):
    input_ids[..., :vocab_size[0]]
    input_ids[..., vocab_size[0]:vocab_size[0]+vocab_size[1]]
    input_ids[..., vocab_size[0]+vocab_size[1]:vocab_size[0]+vocab_size[1]+vocab_size[2]]
    input_ids[..., vocab_size[0]+vocab_size[1]+vocab_size[2]:vocab_size[0]+vocab_size[1]+vocab_size[2]+vocab_size[3]]
    input_ids[..., vocab_size[0]+vocab_size[1]+vocab_size[2]+vocab_size[3]:]
time_2 = time.time() - start


print(f"A: {time_1:.6f} s")
print(f"B: {time_2:.6f} s")
