from __future__ import annotations
from typing import Callable
from functools import partial

import torch
from torch.nn import Module, ModuleList, Linear, RMSNorm, Identity

from einops.layers.torch import Rearrange

from simplicial_attention.simplicial_attention import (
    naive_two_simplicial_attend,
    nth_order_attend
)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# multi-head attention

class HigherOrderAttention(Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,                # query/key head dimension
        dim_head_values = None,       # value head dimension, defaults to `dim_head`
        heads = 8,                    # query heads
        key_value_heads = None,       # key/value heads, default to query heads `heads`       
        number_key_value_sets = 2,    # 2 for 2-simplicial, but can go higher. the century is young
        qk_rmsnorm = True,            # qk rmsnorm, used in a number of models without issues now. helps with stability
        prenorm = False,              # pre rmsnorm for pre-norm transformer pattern
        postnorm = False,             # post rmsnorm, proven out in alphagenome for even more stability (sandwich norm from some old paper i will find later)
        attend: Callable | None = None,
        head_first_dim = True
    ):
        super().__init__()

        # variables

        self.causal = causal
        self.scale = dim_head ** -0.5

        key_value_heads = default(key_value_heads, heads)

        assert divisible_by(heads, key_value_heads)
        self.query_head_groups = heads // key_value_heads

        dim_head_values = default(dim_head_values, dim_head)

        kv_sets = number_key_value_sets

        # maybe pre norm or post norm

        self.prenorm = RMSNorm(dim) if prenorm else Identity()

        self.postnorm = RMSNorm(dim) if postnorm else Identity()

        # maybe qk rmsnorm

        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_norm = RMSNorm(dim_head)
            self.k_norms = ModuleList([RMSNorm(dim_head) for _ in range(kv_sets)])
            self.v_norms = ModuleList([RMSNorm(dim_head) for _ in range(kv_sets)])

        # to queries and sets of keys / values

        self.split_dims = (
            heads * dim_head,                               # queries
            kv_sets * key_value_heads * dim_head,           # keys
            kv_sets * key_value_heads * dim_head_values     # values
        )

        split_heads_eq = 'b n (h d) -> b h n d' if head_first_dim else 'b n (h d) -> b n h d'
        merge_heads_eq = 'b h n d -> b n (h d)' if head_first_dim else 'b n h d -> b n (h d)'

        self.split_q_heads = Rearrange(split_heads_eq, h = heads)
        self.split_kv_heads = Rearrange(split_heads_eq, h = key_value_heads)

        self.kv_sets = kv_sets
        self.to_qkv = Linear(dim, sum(self.split_dims), bias = False)

        # attention function

        self.use_nth_order_attend = kv_sets > 2
        assert not (causal and self.use_nth_order_attend)

        if not exists(attend):
            attend = naive_two_simplicial_attend if not self.use_nth_order_attend else nth_order_attend

        if causal:
            attend = partial(attend, causal = causal)

        self.attend = attend

        # combine heads out

        self.merge_heads = Rearrange(merge_heads_eq)
        self.combine_heads = Linear(heads * dim_head_values, dim, bias = False)

    def forward(
        self,
        tokens
    ):
        tokens = self.prenorm(tokens)

        q, k, v = self.to_qkv(tokens).split(self.split_dims, dim = -1)

        queries = self.split_q_heads(q)
        keys = self.split_kv_heads(k).chunk(self.kv_sets, dim = -1)
        values = self.split_kv_heads(v).chunk(self.kv_sets, dim = -1)

        # maybe qk rmsnorm

        if self.qk_rmsnorm:
            queries = self.q_norm(queries)
            keys = tuple(norm(t) for norm, t in zip(self.k_norms, keys))
            values = tuple(norm(t) for norm, t in zip(self.v_norms, values))

        # higher order attention

        out = self.attend(queries, keys, values)

        # merge heads and combine with linear out

        out = self.merge_heads(out)
        out = self.combine_heads(out)

        return self.postnorm(out)

# 2-simplicial mha

class TwoSimplicialMHA(HigherOrderAttention):
    def __init__(self, *args, **kwargs):
        assert 'number_key_value_sets' not in kwargs

        super().__init__(
            *args,
            number_key_value_sets = 2,
            **kwargs
        )
