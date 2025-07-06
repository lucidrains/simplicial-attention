import torch
from torch import nn

from einops import einsum, rearrange, pack, unpack
from opt_einsum import contract

# functions

def join(arr, delimiter = ', '):
    return delimiter.join(arr)

# 2-simplicial attention

def naive_two_simplicial_attend(
    q,  # b h i d
    k1, # b h j d
    k2, # b h k d
    v1, # b h j dv
    v2  # b h k dv
): # b h i dv

    scale = q.shape[-1] ** -0.5

    q = q * scale

    sim = contract('... i d, ... j d, ... k d -> ... i j k', q, k1, k2)

    packed_sim, packed_shape = pack((sim,), 'b h i *')

    packed_attn = packed_sim.softmax(dim = -1)

    attn, = unpack(packed_attn, packed_shape, 'b h i *')

    out = contract('... i j k, ... j d, ... k d -> ... i d', attn, v1, v2)

    return out

# n-th order attention, for good measure

def nth_order_attend(
    q,     # b h i d
    keys,  # list[b h jkl... d]
    values # list[b h jkl... dv]
):  # b h i dv 

    assert len(keys) == len(values)
    n = len(keys)

    scale = q.shape[-1] * -0.5

    q = q * scale

    # construct equations

    start_index = ord('j')

    ord_indices = list(range(start_index, start_index + n))

    similarity_lfs_eq = join([f'... {chr(i)} d' for i in ord_indices], ', ')

    similarity_rhs_eq = join([chr(i) for i in ord_indices],  ' ')

    similarity_ein_equation = f'... i d, {similarity_lfs_eq} -> ... i {similarity_rhs_eq}'

    aggregate_ein_equation = f'... i {similarity_rhs_eq}, {similarity_lfs_eq} -> ... i d'

    # nth order attention

    sim = einsum(q, *keys, similarity_ein_equation)

    packed_sim, packed_shape = pack((sim,), 'b h i *')

    packed_attn = packed_sim.softmax(dim = -1)

    attn, = unpack(packed_attn, packed_shape, 'b h i *')

    out = einsum(attn, *values, aggregate_ein_equation)

    return out
