
import torch

def test_nth_directional_attn():

    from simplicial_attention.nth_directional_attn import tri_directional_attend

    qk1 = torch.randn(1, 2, 4, 32)
    v1  = torch.randn(1, 2, 4, 32)

    qk2 = torch.randn(1, 2, 8, 32)
    v2  = torch.randn(1, 2, 8, 32)

    qk3 = torch.randn(1, 2, 3, 32)
    v3  = torch.randn(1, 2, 3, 32)

    o1, o2, o3 = tri_directional_attend(
        qk1, v1,
        qk2, v2,
        qk3, v3,
    )

    assert o1.shape == qk1.shape
    assert o2.shape == qk2.shape
    assert o3.shape == qk3.shape
