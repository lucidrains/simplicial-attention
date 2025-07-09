from __future__ import annotations

import torch
from torch import nn, cat, stack, tensor, Tensor

from einops import einsum, rearrange, pack, unpack
from opt_einsum import contract

# functions

def divisible_by(num, den):
    return (num % den) == 0

def join(arr, delimiter = ', '):
    return delimiter.join(arr)

# rotary

def apply_rotation(
    t: Tensor,
    rot: Tensor # Float[3, 3]
):
    device, dim = t.device, t.shape[-1]
    dim = dim // 3 * 3

    t, t_rest = t[..., :dim], t[..., dim:]
    t = rearrange(t, '... (d r) -> ... d r', r = 3)
    t = t @ rot
    t = rearrange(t, '... d r -> ... (d r)')

    return cat((t, t_rest), dim = -1)

# signed determinant

def signed_determinant(q, k1, k2):
    device, dim = q.device, q.shape[-1]
    dim = dim // 3 * 3

    q, q_rest = q[..., :dim], q[..., dim:]
    k1, k1_rest = k1[..., :dim], k1[..., dim:]
    k2, k2_rest = k2[..., :dim], k2[..., dim:]

    has_rest = q_rest.numel() > 0

    # following eq 8.
    # they use this in place of dot product for similarity in attention
    # for rotating in positions and keeping invariance
    # i don't know if all this effort really adds anything, but it is a fun exercise

    k1 = rearrange(k1, '... (d r) -> ... d r', r = 3)
    k2 = rearrange(k2, '... (d r) -> ... d r', r = 3)

    index1 = tensor([2, 0, 1], device = device)
    index2 = tensor([1, 2, 0], device = device)

    lq = q
    rq = q
    lk1 = torch.index_select(k1, dim = -1, index = index2)
    rk1 = torch.index_select(k1, dim = -1, index = index1)
    lk2 = torch.index_select(k2, dim = -1, index = index1)
    rk2 = torch.index_select(k2, dim = -1, index = index2)

    lk1, rk1, lk2, rk2 = tuple(rearrange(t, '... d r -> ... (d r)') for t in (lk1, rk1, lk2, rk2))

    if has_rest:
        lq = cat((lq, q_rest), dim = -1)
        lk1 = cat((lk1, k1_rest), dim = -1)
        lk2 = cat((lk2, k2_rest), dim = -1)

    lhs = einsum(lq, lk1, lk2, 'b h ... i d, b h j d, b h k d -> b h ... i j k')

    rhs = einsum(rq, rk1, rk2, 'b h ... i d, b h j d, b h k d -> b h ... i j k')

    return lhs - rhs

# 2-simplicial attention

def naive_two_simplicial_attend(
    q: Tensor,                  # b h i d
    k: tuple[Tensor, Tensor],   # (b h j d,  b h k d)
    v: tuple[Tensor, Tensor],   # (b h j dv, b h k dv)
    causal = False,
    use_signed_determinant = False
): # b h i dv

    assert len(k) == len(v) == 2

    k1, k2 = k
    v1, v2 = v

    heads, seq_len, dim, kv_heads, device = *q.shape[1:], k1.shape[1], q.device

    assert divisible_by(heads, kv_heads)

    # handle gqa

    groups = heads // kv_heads
    q = rearrange(q, 'b (h g) i d -> b h g i d', g = groups)

    # variables

    scale = dim ** -0.5

    q = q * scale

    if use_signed_determinant:
        sim = signed_determinant(q, k1, k2)
    else:
        sim = contract('... g i d, ... j d, ... k d -> ... g i j k', q, k1, k2)

    if causal:
        i, j = sim.shape[-2:]
        assert i == j

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
    q: Tensor,                  # b h i d
    keys: tuple[Tensor, ...],   # tuple[b h jkl... d]
    values: tuple[Tensor, ...], # tuple[b h jkl... dv]
    causal = False
):  # b h i dv 

    assert len(keys) == len(values)
    n = len(keys)

    heads, seq_len, dim, kv_heads, device = *q.shape[1:], keys[0].shape[1], q.device

    assert divisible_by(heads, kv_heads)

    # handle gqa

    groups = heads // kv_heads
    q = rearrange(q, 'b (h g) i d -> b h g i d', g = groups)

    scale = q.shape[-1] ** -0.5

    q = q * scale

    # construct equations

    start_index = ord('j')

    ord_indices = list(range(start_index, start_index + n))

    similarity_lfs_eq = join([f'... {chr(i)} d' for i in ord_indices], ', ')

    similarity_rhs_eq = join([chr(i) for i in ord_indices],  ' ')

    similarity_ein_equation = f'... g i d, {similarity_lfs_eq} -> ... g i {similarity_rhs_eq}'

    aggregate_ein_equation = f'... g i {similarity_rhs_eq}, {similarity_lfs_eq} -> ... g i d'

    # nth order attention

    sim = contract(similarity_ein_equation, q, *keys)

    # maybe causal

    if causal:
        seq_len = sim.shape[-1]
        one_mask = torch.ones((seq_len, seq_len), device = device, dtype = torch.bool).triu(1)

        causal_mask = one_mask

        for _ in range(n - 1):
            one_mask = one_mask[..., None, :]
            causal_mask = causal_mask[..., :, None] | one_mask

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

    # attention

    packed_sim, packed_shape = pack((sim,), 'b h g i *')

    packed_attn = packed_sim.softmax(dim = -1)

    attn, = unpack(packed_attn, packed_shape, 'b h g i *')

    # aggregate out

    out = contract(aggregate_ein_equation, attn, *values)

    return rearrange(out, 'b h g ... -> b (h g) ...')
