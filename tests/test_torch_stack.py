import torch
import torch.nn as nn
import time

# 参数设置
batch_size = 32
seq_len = 128
embed_dim = 64
num_categories = 5
vocab_sizes = [128] * num_categories

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模拟输入
input_ids = torch.randint(0, 128, (batch_size, seq_len, num_categories))

# embedding ModuleList
embeds = nn.ModuleList([nn.Embedding(siz, embed_dim) for siz in vocab_sizes])

# 方法 A: stack + mean
start = time.time()
for _ in range(100):
    x = torch.stack([embeds[i](input_ids[..., i]) for i in range(num_categories)], dim=0).mean(dim=0)
torch.cuda.synchronize() if torch.cuda.is_available() else None
time_stack = time.time() - start

# 方法 B: sum / len
start = time.time()
for _ in range(100):
    x = sum(embeds[i](input_ids[..., i]) for i in range(num_categories)) / num_categories
torch.cuda.synchronize() if torch.cuda.is_available() else None
time_sum = time.time() - start

x = torch.zeros((batch_size, seq_len, embed_dim), device=input_ids.device)
for _ in range(100):
    for i in range(num_categories):
        x += embeds[i](input_ids[..., i])
    x /= num_categories
torch.cuda.synchronize() if torch.cuda.is_available() else None
time_sum_2 = time.time() - start

print(f"Stack + mean: {time_stack:.6f} s")
print(f"Sum / len:   {time_sum:.6f} s")
print(f"inplace:   {time_sum_2:.6f} s")
