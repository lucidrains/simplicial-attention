from __future__ import annotations

import torch
from torch import nn, Tensor

from einops import einsum, rearrange, pack, unpack
from opt_einsum import contract

# functions

def divisible_by(num, den):
    return (num % den) == 0

def join(arr, delimiter = ', '):
    return delimiter.join(arr)

# 2-simplicial attention

def naive_two_simplicial_attend(
    q,  # b h i d
    k1, # b h j d
    k2, # b h k d
    v1, # b h j dv
    v2, # b h k dv,
    causal = False
): # b h i dv

    heads, seq_len, dim, kv_heads, device = *q.shape[1:], k1.shape[1], q.device

    assert divisible_by(heads, kv_heads)

    # handle gqa

    groups = heads // kv_heads
    q = rearrange(q, 'b (h g) i d -> b h g i d', g = groups)

    # variables

    scale = dim ** -0.5

    q = q * scale

    sim = contract('... g i d, ... j d, ... k d -> ... g i j k', q, k1, k2)

    if causal:
        i, j = sim.shape[-2:]
        causal_mask = torch.ones(i, j, device = device, dtype = torch.bool).triu(j - i + 1)
        causal_mask = causal_mask[..., :, None] | causal_mask[..., None, :]
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

    packed_sim, packed_shape = pack((sim,), 'b h g i *')

    packed_attn = packed_sim.softmax(dim = -1)

    attn, = unpack(packed_attn, packed_shape, 'b h g i *')

    out = contract('... g i j k, ... j d, ... k d -> ... g i d', attn, v1, v2)

    return rearrange(out, 'b h g ... -> b (h g) ...')

# n-th order attention, for good measure

def nth_order_attend(
    q,                   # b h i d
    keys: list[Tensor],  # list[b h jkl... d]
    values: list[Tensor] # list[b h jkl... dv]
):  # b h i dv 

    assert len(keys) == len(values)
    n = len(keys)

    heads, seq_len, dim, kv_heads, device = *q.shape[1:], keys[0].shape[1], q.device

    assert divisible_by(heads, kv_heads)

    # handle gqa

    groups = heads // kv_heads
    q = rearrange(q, 'b (h g) i d -> b h g i d', g = groups)

    scale = q.shape[-1] * -0.5

    q = q * scale

    # construct equations

    start_index = ord('j')

    ord_indices = list(range(start_index, start_index + n))

    similarity_lfs_eq = join([f'... {chr(i)} d' for i in ord_indices], ', ')

    similarity_rhs_eq = join([chr(i) for i in ord_indices],  ' ')

    similarity_ein_equation = f'... g i d, {similarity_lfs_eq} -> ... g i {similarity_rhs_eq}'

    aggregate_ein_equation = f'... g i {similarity_rhs_eq}, {similarity_lfs_eq} -> ... g i d'

    # nth order attention

    sim = einsum(q, *keys, similarity_ein_equation)

    packed_sim, packed_shape = pack((sim,), 'b h g i *')

    packed_attn = packed_sim.softmax(dim = -1)

    attn, = unpack(packed_attn, packed_shape, 'b h g i *')

    out = einsum(attn, *values, aggregate_ein_equation)

    return rearrange(out, 'b h g ... -> b (h g) ...')
