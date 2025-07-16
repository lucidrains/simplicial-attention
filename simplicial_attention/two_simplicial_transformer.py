from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat

from simplicial_attention.triton_two_simplicial_attention import SlidingWindowTwoSimplicialMHA

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# classes

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        key_value_heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.key_value_heads = key_value_heads
        assert divisible_by(heads, key_value_heads)
        self.groups = heads // key_value_heads

        dim_inner = heads * dim_head
        dim_kv_inner = key_value_heads * dim_head

        self.prenorm = nn.RMSNorm(dim)
        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_kv = nn.Linear(dim, dim_kv_inner * 2, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x
    ):
        x = self.prenorm(x)

        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h = self.heads)
        k, v = repeat(self.to_kv(x), 'b n (kv h d) -> kv b (h g) n d', h = self.key_value_heads, kv = 2, g = self.groups)

        # attention branch

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True
        )

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

# simple feedforward

def FeedForward(dim, expansion = 4.):
    dim_inner = int(dim * expansion)
    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# transformer

class TwoSimplicialTransformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        heads = 8,
        key_value_heads = 4,
        dim_head = 64,
        ff_expansion = 4.,
        two_simplicial_attn_every = 4, # they do every 4 layers
        attn_kwargs: dict = dict(),
        two_simplicial_attn_kwargs: dict = dict(
            w1 = 512,
            w2 = 32,
            qk_rmsnorm = True
        ),
        final_norm = True
    ):
        super().__init__()

        layers = []

        for layer_index in range(depth):

            use_higher_order_attn = divisible_by(layer_index + 1, two_simplicial_attn_every)

            if use_higher_order_attn:
                attn = SlidingWindowTwoSimplicialMHA(
                    dim = dim,
                    heads = heads,
                    key_value_heads = key_value_heads,
                    dim_head = dim_head,
                    prenorm = True,
                    **two_simplicial_attn_kwargs
                )
            else:
                attn = Attention(
                    dim = dim,
                    heads = heads,
                    key_value_heads = key_value_heads,
                    dim_head = dim_head,
                    **attn_kwargs
                )

            ff = FeedForward(dim = dim, expansion = ff_expansion)

            layers.append(ModuleList([
                attn,
                ff
            ]))

        self.layers = ModuleList(layers)
        self.norm = nn.RMSNorm(dim) if final_norm else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
