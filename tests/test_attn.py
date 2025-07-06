import pytest
import torch

def test_attn():
    from simplicial_attention.simplicial_attention import naive_two_simplicial_attend

    q = torch.randn(1, 8, 32, 16)
    k = torch.randn(1, 8, 32, 16)
    v = torch.randn(1, 8, 32, 16)

    attended = naive_two_simplicial_attend(
        q,
        k,
        k.clone(),
        v,
        v.clone()
    )

    assert attended.shape == q.shape

def test_fifth_order():
    from simplicial_attention import nth_order_attend

    q = torch.randn(1, 8, 4, 16)
    k = torch.randn(1, 8, 4, 16)
    v = torch.randn(1, 8, 4, 16)

    fifth_order_attended = nth_order_attend(
        q,
        [k, k, k, k, k],
        [v, v, v, v, v]
    )

    assert fifth_order_attended.shape == q.shape
