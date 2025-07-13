
import pytest
import torch

@pytest.mark.parametrize('has_mask', (False, True))
def test_tridirectional_attend(has_mask):

    from simplicial_attention.nth_directional_attn import tri_directional_attend

    qk1 = torch.randn(1, 2, 4, 32)
    v1  = torch.randn(1, 2, 4, 32)

    qk2 = torch.randn(1, 2, 8, 32)
    v2  = torch.randn(1, 2, 8, 32)

    qk3 = torch.randn(1, 2, 3, 32)
    v3  = torch.randn(1, 2, 3, 32)

    mask1 = mask2 = mask3 = None

    if has_mask:
        mask1 = torch.randint(0, 2, (1, 4)).bool()
        mask2 = torch.randint(0, 2, (1, 8)).bool()
        mask3 = torch.randint(0, 2, (1, 3)).bool()

    o1, o2, o3 = tri_directional_attend(
        qk1, v1,
        qk2, v2,
        qk3, v3,
        mask1, mask2, mask3
    )

    assert o1.shape == qk1.shape
    assert o2.shape == qk2.shape
    assert o3.shape == qk3.shape


def test_nthdirectional_attend():

    from simplicial_attention.nth_directional_attn import nth_directional_attend

    qk1 = torch.randn(1, 2, 4, 32)
    v1  = torch.randn(1, 2, 4, 32)

    qk2 = torch.randn(1, 2, 8, 32)
    v2  = torch.randn(1, 2, 8, 32)

    qk3 = torch.randn(1, 2, 3, 32)
    v3  = torch.randn(1, 2, 3, 32)

    qk4 = torch.randn(1, 2, 7, 32)
    v4 = torch.randn(1, 2, 7, 32)

    o1, o2, o3, o4 = nth_directional_attend(
        (qk1, v1),
        (qk2, v2),
        (qk3, v3),
        (qk4, v4)
    )

    assert o1.shape == qk1.shape
    assert o2.shape == qk2.shape
    assert o3.shape == qk3.shape
    assert o4.shape == qk4.shape
