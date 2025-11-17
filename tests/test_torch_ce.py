import torch
import torch.nn.functional as F
import time

torch.device("cuda" if torch.cuda.is_available() else "cpu")
def multihead_ce_single_kernel(logits, targets, vocab_sizes, pad_token_id):
    # logits: (B, T, total_vocab)
    # targets: (B, T, F)
    B, T, Vtotal = logits.shape
    # F = len(vocab_sizes)
    
    # build slices
    slices = []
    offset = 0
    for v in vocab_sizes:
        slices.append((offset, offset+v))
        offset += v
    
    # 1. slice & pad vocab
    Vmax = max(vocab_sizes)

    padded_logits = []
    masks = []

    for (start, end), v in zip(slices, vocab_sizes):
        lg = logits[..., start:end]   # (B,T,v)
        pad_len = Vmax - v

        if pad_len > 0:
            lg = F.pad(lg, (0, pad_len))

        padded_logits.append(lg)

        m = torch.zeros(Vmax, dtype=torch.bool, device=logits.device)
        m[:v] = True
        masks.append(m)

    logits_4d = torch.stack(padded_logits, dim=2)   # (B,T,F,Vmax)
    mask_4d = torch.stack(masks, dim=0).view(1,1,len(vocab_sizes),Vmax)

    masked_logits = logits_4d.masked_fill(~mask_4d, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=-1)

    # 2. gather
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # 3. mask padded labels
    valid = (targets != pad_token_id)

    return (nll * valid).sum() / valid.sum()


def benchmark(fn, name, n=2500):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        logits = torch.randn(B, T, total_vocab, device="cuda")
        targets = torch.randint(2, 30, (B, T, Fdim), device="cuda")
        loss = fn()
        # loss.backward()
    torch.cuda.synchronize()
    print(f"{name:20s} {time.time()-t0:.5f}s")
    time.sleep(2)

    


# --------- Mock config ---------
B = 256
T = 512
Fdim = 5
vocab_sizes = [64, 32, 88, 99, 256]  # example
pad_id = 0
total_vocab = sum(vocab_sizes)

logits = torch.randn(B, T, total_vocab, device="cuda")
targets = torch.randint(2, 30, (B, T, Fdim), device="cuda")

# --------- Method0 ---------
def method0():
    split = logits.split(vocab_sizes, -1)
    loss = 0
    for i, lg in enumerate(split):
        # print(lg.reshape(-1, lg.size(-1)).shape)
        # print(targets[...,i].reshape(-1).shape)
        loss += F.cross_entropy(
            lg.reshape(-1, lg.size(-1)),
            targets[...,i].reshape(-1),
            ignore_index=pad_id
        )
    return loss

# --------- MethodA ---------
def methodA():
    loss = 0
    offset = 0
    flat_targets = targets.view(-1, Fdim)
    for i, v in enumerate(vocab_sizes):
        lg = logits[..., offset:offset+v]
        offset += v
        loss += F.cross_entropy(
            lg.reshape(-1, v),
            flat_targets[:, i],
            ignore_index=pad_id
        )
    return loss

# --------- MethodB ---------
def methodB():
    logits_T = logits.view(-1, total_vocab)
    targets_T = targets.view(-1, Fdim)

    offset = 0
    loss = 0
    for i, v in enumerate(vocab_sizes):
        lg = logits_T[:, offset:offset+v]
        lb = targets_T[:, i]
        offset += v
        loss += F.cross_entropy(lg, lb, ignore_index=pad_id)
    return loss

# --------- MethodC (fastest) ---------
# def methodC():
#     return multihead_ce_single_kernel(logits, targets, vocab_sizes, pad_id)

def methodC(): # Real C
    logits_T = logits.view(-1, total_vocab)
    targets_T = targets.view(-1, Fdim)

    split = logits_T.split(vocab_sizes, dim = -1)

    offset = 0
    loss = 0
    for i, v in enumerate(vocab_sizes):
        lg = split[i]
        lb = targets_T[:, i]
        offset += v
        loss += F.cross_entropy(lg, lb, ignore_index=pad_id)
    return loss



benchmark(method0, "Loop 5 CE")
benchmark(methodA, "Fast slice")
benchmark(methodC, "Single kernel (best)")
benchmark(methodB, "Tensor reshape")

benchmark(method0, "Loop 5 CE")
benchmark(methodA, "Fast slice")
benchmark(methodB, "Tensor reshape")
benchmark(methodC, "Single kernel (best)")

