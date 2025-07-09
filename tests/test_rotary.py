import torch
from torch import sin, cos, stack, tensor, cat
from einops import rearrange, einsum

def test_signed_determinant():
    from simplicial_attention.simplicial_attention import apply_rotation, signed_determinant

    # random rotations

    def rot_z(gamma):
        c = cos(gamma)
        s = sin(gamma)
        z = torch.zeros_like(gamma)
        o = torch.ones_like(gamma)

        out = stack((
            c, -s, z,
            s, c, z,
            z, z, o
        ), dim = -1)

        return rearrange(out, '... (r1 r2) -> ... r1 r2', r1 = 3)

    def rot_y(beta):
        c = cos(beta)
        s = sin(beta)
        z = torch.zeros_like(beta)
        o = torch.ones_like(beta)

        out = stack((
            c, z, s,
            z, o, z,
            -s, z, c
        ), dim = -1)

        return rearrange(out, '... (r1 r2) -> ... r1 r2', r1 = 3)

    def rot(alpha, beta, gamma):
        return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

    R = rot(*torch.randn(3))

    q = torch.randn(1, 8, 4, 7)
    k1 = torch.randn(1, 8, 4, 7)
    k2 = torch.randn(1, 8, 4, 7)

    rq = apply_rotation(q, R)
    rk1 = apply_rotation(k1, R)
    rk2 = apply_rotation(k2, R)

    sim = einsum(q, k1, k2, '... i d, ... j d, ... k d -> ... i j k')
    rsim = einsum(rq, rk1, rk2, '... i d, ... j d, ... k d -> ... i j k')

    assert not torch.allclose(sim, rsim, atol = 1e-5)

    sim = signed_determinant(q, k1, k2)
    rsim = signed_determinant(rq, rk1, rk2)

    assert torch.allclose(sim, rsim, atol = 1e-5)
