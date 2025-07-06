import torch
from torch import nn

from einops import einsum, rearrange, pack, unpack
from opt_einsum import contract

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
