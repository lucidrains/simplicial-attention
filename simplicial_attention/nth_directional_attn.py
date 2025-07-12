import torch
from torch import arange

from einops import einsum, rearrange

# functions

def join(arr, delimiter = ', '):
    return delimiter.join(arr)

# taking the idea from
# https://github.com/lucidrains/bidirectional-cross-attention

# and extending to trilinear (and eventually nth-linear)
# think there is some parallel to the thalamus
# https://www.youtube.com/watch?v=Dykkubb-Qus

def tri_directional_attend(
    qk1, # (b, h, s, d)
    v1,  # (b, h, s, dv)
    qk2, # (b, h, t, d)
    v2,  # (b, h, t, dv)
    qk3, # (b, h, r, d)
    v3,  # (b, h, r, dv)
): # (b h s dv), (b h t dv), (b h r dv)

    device = qk1.device

    scale = qk1.shape[-1] ** -0.5

    sim = einsum(qk1, qk2, qk3, '... i d, ... j d, ... k d -> ... i j k')

    sim = sim * scale

    values = [v1, v2, v3]

    outputs = [] # outputs per one modality accumulating others

    for source_index in range(3):

        # move axis so (batch, heads, source, target1, target2 ..)

        source_sim = sim.moveaxis(2 + source_index, 2)

        # get the values to be aggregated for the axis

        aggregate_values = [v for value_index, v in enumerate(values) if value_index != source_index]

        # softmax

        source_sim = rearrange(source_sim, 'b h n ... -> b h n (...)')
        attn = source_sim.softmax(dim = -1)

        # outer

        acc, *rests = aggregate_values

        for rest in rests:
            acc = acc[..., :, None, :] * rest[..., None, :, :]
            acc = rearrange(acc, '... i j d -> ... (i j) d')

        # aggregate

        out = einsum(attn, acc, 'b h i j, b h j d -> b h i d')

        outputs.append(out)

    return tuple(outputs)

def nth_directional_attend(
    *qkvs, # ((b h s d), (b h s dv)) * num modalities
):
    num_modalities = len(qkvs)

    assert len(qkvs) > 1 and all(len(qkv) == 2 for qkv in qkvs)

    qk1 = qkvs[0][0]

    device = qk1.device

    scale = qk1.shape[-1] ** -0.5

    queries_keys = [qk for qk, v in qkvs]
    values       = [v for _, v in qkvs]

    # get the einsum equation for similariy

    start_index = ord('i')

    ord_indices = list(range(start_index, start_index + num_modalities))

    similarity_lfs_eq = join([f'... {chr(i)} d' for i in ord_indices], ', ')

    similarity_rhs_eq = join([chr(i) for i in ord_indices],  ' ')

    similarity_ein_equation = f'{similarity_lfs_eq} -> ... {similarity_rhs_eq}'

    # similarity

    sim = einsum(*queries_keys, similarity_ein_equation)

    # scale

    sim = sim * scale

    outputs = [] # outputs per one modality accumulating others

    for source_index in range(num_modalities):

        # move axis so (batch, heads, source, target1, target2 ..)

        source_sim = sim.moveaxis(2 + source_index, 2)

        # get the values to be aggregated for the axis

        aggregate_values = [v for value_index, v in enumerate(values) if value_index != source_index]

        # softmax

        source_sim = rearrange(source_sim, 'b h n ... -> b h n (...)')
        attn = source_sim.softmax(dim = -1)

        # outer

        acc, *rests = aggregate_values

        for rest in rests:
            acc = acc[..., :, None, :] * rest[..., None, :, :]
            acc = rearrange(acc, '... i j d -> ... (i j) d')

        # aggregate

        out = einsum(attn, acc, 'b h i j, b h j d -> b h i d')

        outputs.append(out)

    return tuple(outputs)
