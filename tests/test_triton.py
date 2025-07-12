import torch
import pytest

# helpers

def exists(v):
    return v is not None

def test_kernels():

    if not torch.cuda.is_available():
        pytest.skip()

    from einops import rearrange, einsum
    from simplicial_attention.triton_two_simplicial_attention import sliding_two_simplicial_attn

    # flex attention
    # https://pytorch.org/blog/flexattention/

    flex_attention = None

    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        if torch.cuda.is_available():
            flex_attention = torch.compile(flex_attention)
    except ImportError:
        pass

    assert exists(flex_attention)

    def create_tilinear_sliding_mask(seq_len, window1_size, window2_size):

        def sliding_mask(_, __, q_index, kv_index):

            kv1_index = kv_index // seq_len
            kv2_index = kv_index % seq_len

            distance1 = q_index - kv1_index
            distance2 = q_index - kv2_index

            backward_sliding_mask1 = distance1 <= window1_size
            backward_sliding_mask2 = distance2 <= window2_size

            forward_sliding_mask1 = distance1 >= 0
            forward_sliding_mask2 = distance2 >= 0

            return backward_sliding_mask1 & backward_sliding_mask2 & forward_sliding_mask1 & forward_sliding_mask2

        block_mask = create_block_mask(sliding_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len * seq_len, _compile = True)
        return block_mask

    def flex_sliding_two_simplicial_attn(q, k1, k2, v1, v2, w1 = 64, w2 = 32):
        q, k1, k2, v1, v2 = tuple(rearrange(t, 'b n h d -> b h n d') for t in (q, k1, k2, v1, v2))

        seq_len = q.shape[-2]

        k = einsum(k1, k2, 'b h i d, b h j d -> b h i j d')
        v = einsum(v1, v2, 'b h i d, b h j d -> b h i j d')
        k, v = tuple(rearrange(t, 'b h i j d -> b h (i j) d') for t in (k, v))

        block_mask = create_tilinear_sliding_mask(seq_len, w1, w2)

        out = flex_attention(q, k, v, block_mask = block_mask, enable_gqa = True)
        return rearrange(out, 'b h n d -> b n h d')

    # queries, keys, values

    q = torch.randn(2, 1024, 4, 64).cuda()
    k1 = torch.randn(2, 1024, 4, 64).cuda()
    k2 = torch.randn(2, 1024, 4, 64).cuda()
    v1 = torch.randn(2, 1024, 4, 64).cuda()
    v2 = torch.randn(2, 1024, 4, 64).cuda()

    tq = q.detach().clone()
    tk1 = k1.detach().clone()
    tk2 = k2.detach().clone()
    tv1 = v1.detach().clone()
    tv2 = v2.detach().clone()

    w1 = 64
    w2 = 32

    (q, k1, k2, v1, v2, tq, tk1, tk2, tv1, tv2) = tuple(t.requires_grad_() for t in (q, k1, k2, v1, v2, tq, tk1, tk2, tv1, tv2))

    # inefficient way

    flex_forward_out = flex_sliding_two_simplicial_attn(q, k1, k2, v1, v2, w1 = w1, w2 = w2)

    # triton kernel

    triton_forward_out = sliding_two_simplicial_attn(
        tq,
        (tk1, tk2),
        (tv1, tv2),
        w1 = w1,
        w2 = w2
    )

    # asserts

    max_diff = (flex_forward_out - triton_forward_out).abs().amax()
    print(f'forward abs diff: {max_diff.item():.3f}')

    assert torch.allclose(flex_forward_out, triton_forward_out, atol = 1e-2), 'output not equal'

    # backwards

    flex_forward_out.sum().backward()
    triton_forward_out.sum().backward()

    assert torch.allclose(v1.grad, tv1.grad, atol = 1e-2), 'v1 grad not equal'
    assert torch.allclose(v2.grad, tv2.grad, atol = 1e-2), 'v2 grad not equal'
    assert torch.allclose(k1.grad, tk1.grad, atol = 1e-2), 'k1 grad not equal'
    assert torch.allclose(k2.grad, tk2.grad, atol = 1e-2), 'k2 grad not equal'
    assert torch.allclose(q.grad, tq.grad, atol = 1e-2), 'q grad not equal'
