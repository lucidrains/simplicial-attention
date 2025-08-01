import math
import gzip
import random
import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from einops import rearrange
from simplicial_attention.two_simplicial_transformer import TwoSimplicialTransformer

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 32
GENERATE_EVERY = 500
GENERATE_LENGTH = 256
SEQ_LEN = 256

# helpers

def exists(v):
    return v is not None

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

class TwoSimplicialLM(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len = SEQ_LEN,
        dim_head = 64,
        heads = 8,
        two_simplicial_attn_every = 4,
        transformer_kwargs = dict()
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.transformer = TwoSimplicialTransformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            two_simplicial_attn_every = two_simplicial_attn_every,
            **transformer_kwargs
        )

        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def sample(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.,
        filter_thres = 0.9,
    ):
        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        for _ in range(sample_num_times):
            logits = self.forward(out, return_loss = False)
            logits = logits[:, -1]

            logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(logits, temperature = temperature, dim = -1)

            out = torch.cat((out, sample), dim = -1)

        return out[..., prompt_seq_len:]

    def forward(self, x, return_loss = False):

        if return_loss:
            x, target = x[:, :-1], x[:, 1:]

        seq_len, device = x.shape[-1], x.device

        tokens = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_len, device = device))

        tokens = tokens + pos_emb

        embed = self.transformer(tokens)

        logits = self.to_logits(embed)

        if not return_loss:
            return logits

        return F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            target
        )

model = TwoSimplicialLM(
    num_tokens = 256,
    dim = 512,
    depth = 6,
    two_simplicial_attn_every = 2 # every 2 layers
).cuda()

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# optimizer

optim = Adam(model.parameters(), lr = LEARNING_RATE)

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model.train()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        loss = model(data, return_loss = True)

        (loss / GRAD_ACCUM_EVERY).backward()

    print(f"training loss: {loss.item():.3f}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)

            loss = model(valid_data, return_loss = True)
            print(f"validation loss: {loss.item():.3f}")

    if i % GENERATE_EVERY == 0:
        model.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        inp = inp.cuda()

        prime = decode_tokens(inp)
        print(f"\n\nINPUT: {prime}")

        prompt = inp[None, ...]

        sampled = model.sample(prompt, GENERATE_LENGTH)

        base_decode_output = decode_tokens(sampled[0])

        print(f"\nOUTPUT: {base_decode_output}\n\n")
