from .sparse import *
import torch
from .semirings import LogSemiring

def test_sparse():
    r = torch.rand(20, 10, 10)
    q = sparse_to_dense(dense_to_sparse(r, 5, offset=-2), offset=-2)

    assert torch.isclose(r, q).diagonal(-1, 1, 2).all()
    assert torch.isclose(r, q).diagonal(-2, 1, 2).all()
    assert torch.isclose(r, q).diagonal(0, 1, 2).all()

    q = sparse_to_dense(dense_to_sparse(r, 5, offset=2), offset=2)

    assert torch.isclose(r, q).diagonal(1, 1, 2).all()
    assert torch.isclose(r, q).diagonal(2, 1, 2).all()
    assert torch.isclose(r, q).diagonal(0, 1, 2).all()

    # Mat Mul
    sparse = dense_to_sparse(r.transpose(-2, -1), 5)
    o = sparse_to_dense(sparse)

    a = dense_to_sparse(r.transpose(-2, -1), 5)
    b = dense_to_sparse(r, 5)
    assert torch.isclose(sparse_to_dense(sparse_combine(a, b))[0],
                         torch.matmul(o, o)[0]).all()


    r1 = torch.rand(3, 3)
    r2 = torch.rand(3, 3)
    a = dense_to_sparse(r1.transpose(-2, -1), 5)
    b = dense_to_sparse(r2, 5)


    # print(torch.matmul(r1, r2))
    # print(sparse_to_dense(sparse_combine(b, a)))
    combo = sparse_combine(b, a)
    print(combo)
    assert torch.isclose(sparse_to_dense(combo),
                         torch.matmul(r1, r2)).all()

    r1 = torch.rand(10, 10)
    r2 = torch.rand(10, 10)

    a = dense_to_sparse(r1.transpose(-2, -1), 5)
    b = dense_to_sparse(r2, 5)
    c = sparse_combine(b, a)

    a = dense_to_sparse(r1.transpose(-2, -1), 5)
    b = dense_to_sparse(r2, 5)
    c2 = sparse_combine(a, b)
    c2 = flip(c2, 9)

    assert torch.isclose(c, c2).all()

    r1 = torch.rand(1, 10, 10)
    r2 = torch.rand(5, 10, 10)

    r1 = torch.rand(1, 4, 4)
    r2 = torch.rand(5, 4, 4)


    a = dense_to_sparse(r1.transpose(-2, -1), 5, semiring=LogSemiring)
    b = dense_to_sparse(r2, 5, semiring=LogSemiring)
    c = sparse_combine(b, a, semiring=LogSemiring,fn=lambda a, b: torch.logsumexp(a + b, dim=-1))

    a = dense_to_sparse(r1.transpose(-2, -1), 5, semiring=LogSemiring)
    b = dense_to_sparse(r2, 5, semiring=LogSemiring)
    c2 = sparse_combine(a, b,semiring=LogSemiring, fn=lambda a, b: torch.logsumexp(a + b, dim=-1))
    c2 = flip(c2, 9, semiring=LogSemiring)


    print((c.exp() - c2.exp()))
    assert torch.isclose(c.exp(), c2.exp()).all()


    a = dense_to_sparse(r1.transpose(-2, -1), 5, offset= 1, semiring=LogSemiring)
    b = dense_to_sparse(r2, 5, offset= 0, semiring=LogSemiring)
    c = sparse_combine(b, a, semiring=LogSemiring,fn=lambda a, b: torch.logsumexp(a + b, dim=-1))

    a = dense_to_sparse(r1.transpose(-2, -1), 5, offset= 1, semiring=LogSemiring)
    b = dense_to_sparse(r2, 5, offset= 0, semiring=LogSemiring)
    c2 = sparse_combine(a, b,semiring=LogSemiring, fn=lambda a, b: torch.logsumexp(a + b, dim=-1))
    c2 = flip(c2, 9, semiring=LogSemiring)

    print((c.exp() - c2.exp()))
    assert torch.isclose(c.exp(), c2.exp()).all()

def test_flip():
    r1 = torch.rand(1, 10, 10)
    r2 = torch.rand(5, 10, 10)
    a = dense_to_sparse(r1.transpose(-2, -1), 5, offset= 0, semiring=LogSemiring)
    b = dense_to_sparse(r2, 5, offset= 0, semiring=LogSemiring)
    c = sparse_combine(b, a, semiring=LogSemiring,fn=lambda a, b: torch.logsumexp(a + b, dim=-1))


    a = dense_to_sparse(r1, 5, offset= 0, semiring=LogSemiring)
    a = flip(a, 5, semiring=LogSemiring)
    b = dense_to_sparse(r2, 5, offset= 0, semiring=LogSemiring)
    c2 = sparse_combine(b, a, semiring=LogSemiring,fn=lambda a, b: torch.logsumexp(a + b, dim=-1))
    assert torch.isclose(c.exp(), c2.exp()).all()

def test_flip_offset():
    r1 = torch.rand(1, 5, 5)
    r2 = torch.rand(5, 5, 5)
    r1 = sparse_to_dense(dense_to_sparse(r1, 5, semiring=LogSemiring), semiring=LogSemiring)
    r2 = sparse_to_dense(dense_to_sparse(r2, 5, semiring=LogSemiring), semiring=LogSemiring)

    a = dense_to_sparse(r1.transpose(-2, -1), 5, offset= 1, semiring=LogSemiring)
    b = dense_to_sparse(r2, 5, offset= 0, semiring=LogSemiring)
    c = sparse_combine(b, a, semiring=LogSemiring,fn=lambda a, b: torch.logsumexp(a + b, dim=-1))


    a = dense_to_sparse(r1, 5, offset= 0, semiring=LogSemiring)
    a = flip(a, 5, semiring=LogSemiring)
    a = pad(a, 0, -1, offset=1, semiring=LogSemiring)
    b = dense_to_sparse(r2, 5, offset= 0, semiring=LogSemiring)

    c2 = sparse_combine(b, a,
                        semiring=LogSemiring,fn=lambda a, b: torch.logsumexp(a + b, dim=-1))
    assert torch.isclose(c.exp(), c2.exp()).all()

    r1 = torch.rand(1, 10, 10)
    r2 = torch.rand(1, 10, 10)
    r1 = sparse_to_dense(dense_to_sparse(r1, 5, semiring=LogSemiring), semiring=LogSemiring)
    r2 = sparse_to_dense(dense_to_sparse(r2, 5, semiring=LogSemiring), semiring=LogSemiring)

    r3 = LogSemiring.dot(r1.unsqueeze(-2), r2.transpose(-2, -1).unsqueeze(-3))


    a = dense_to_sparse(r1.transpose(-2, -1), 5, offset=0, semiring=LogSemiring)
    b = dense_to_sparse(r2, 5, offset= 1, semiring=LogSemiring)
    c = sparse_combine(b, a, semiring=LogSemiring,fn=lambda a, b: torch.logsumexp(a + b, dim=-1))

    a = dense_to_sparse(r1, 5, offset= 0, semiring=LogSemiring)
    a = flip(a, 5, semiring=LogSemiring)

    b = dense_to_sparse(r2, 5, offset= 0, semiring=LogSemiring)
    b = pad(b, 0, -1, offset=1, semiring=LogSemiring)

    c2 = sparse_combine(b, a,
                        semiring=LogSemiring,fn=lambda a, b: torch.logsumexp(a + b, dim=-1))
    assert torch.isclose(c.exp(), c2.exp()).all()


    # Check
    a = dense_to_sparse(r1.transpose(-2, -1), 5, offset=0, semiring=LogSemiring)
    b = dense_to_sparse(r2, 5, offset=0, semiring=LogSemiring)
    c = sparse_combine(b, a, semiring=LogSemiring,fn=lambda a, b: torch.logsumexp(a + b, dim=-1))

    c3 = dense_to_sparse(r3, 9, semiring=LogSemiring)
    assert torch.isclose(c.exp(), c3.exp()).all()


    # Compare to dense offset
    start = 3
    end = 5
    r1 = sparse_to_dense(dense_to_sparse(r1, start, semiring=LogSemiring), semiring=LogSemiring)
    r2 = sparse_to_dense(dense_to_sparse(r2, start, semiring=LogSemiring), semiring=LogSemiring)
    r4 = LogSemiring.dot(r1[...,1:].unsqueeze(-2),
                         r2.transpose(-2, -1)[...,:-1].unsqueeze(-3))


    a = dense_to_sparse(r1, start, offset= 0, semiring=LogSemiring)

    a_check = bot_pad(a[..., 1:, :], 1, -2, semiring=LogSemiring)
    dense_left = bot_pad(r1[..., 1:], 1, -1, semiring=LogSemiring)
    comp_left = dense_to_sparse(dense_left,
                                start, offset=-1, semiring=LogSemiring)
    # q = flip(a_check, start, semiring=LogSemiring)
    # print(q.exp())
    assert torch.isclose(a_check, comp_left).all()

    a_check2 = bot_pad(a, 1, -2, semiring=LogSemiring)
    a2 = flip(a_check2,#top_pad(a_check, 1, -2, semiring=LogSemiring),
              start, semiring=LogSemiring)
    back = flip(a2, start, semiring=LogSemiring)
    assert torch.isclose(back[..., 1:, :].exp(), a_check.exp()).all()
    # a = a2[..., 1:, :]
    a = a2
    # a = bot_pad(a2[..., 1:, :], 1, -2, semiring=LogSemiring)
    # print(a.exp())

    b = dense_to_sparse(r2, start, offset= 0, semiring=LogSemiring)
    dense_right = bot_pad(r2[...,:-1], 1, -1, semiring=LogSemiring)
    comp_right = dense_to_sparse(dense_right,
                                 start, semiring=LogSemiring)
    b_check = bot_pad(b[..., :-1, :], 1, -2, semiring=LogSemiring)

    # print("b")
    # print(b.exp())
    # print(comp_right.exp())
    assert torch.isclose(b_check, comp_right).all()


    # b = b
    b = top_pad(b, 1, -2, semiring=LogSemiring)

    # print("final", a.shape, b.shape)
    # print(r1[..., 1:].exp())
    # print(a.exp())
    # print(a2.exp())
    # print(a_check.exp())
    # print(r2[..., :-1].exp())
    # print(b.exp())

    c2 = sparse_combine(b, a,
                        semiring=LogSemiring,
                        fn=lambda a, b: torch.logsumexp(a + b, dim=-1))
    print(a[..., 0, :])
    print(b[..., 0, :])
    print(c2[..., 0, :])

    c2 = c2[..., 1:, :]
    c3 = dense_to_sparse(r4, end, offset=-1, semiring=LogSemiring)
    # print(c2[0].exp())
    # print(c3[0].exp())
    # print(c2[0].exp() -
    #       c3[0].exp())

    assert torch.isclose(c2.exp(), c3.exp()).all()


    # FINISH

    # flippeda = pad(b.flip(1), 4, 0).unfold(0, 5, 1).diagonal(0, 1, 2)
    # flippedb = pad(b.flip(1), 4, 0).unfold(0, 5, 1).diagonal(0, 1, 2)

    # combo = sparse_combine(a, b, back=True)
    # print(combo)
    # flipped = pad(combo.flip(1), 8, 0).unfold(0, 9, 1).diagonal(0, 1, 2)


    # assert torch.isclose(sparse_to_dense(flipped),
    #                      torch.matmul(r1, r2)).all()

def test_pad():
    r1 = torch.rand(1, 10, 10)
    r2 = torch.rand(1, 10, 10)
    start = 3
    end = 5
    r1 = sparse_to_dense(dense_to_sparse(r1, start, semiring=LogSemiring), semiring=LogSemiring)
    r2 = sparse_to_dense(dense_to_sparse(r2, start, semiring=LogSemiring), semiring=LogSemiring)
    r4 = LogSemiring.dot(r1[...,1:].unsqueeze(-2),
                         r2.transpose(-2, -1)[...,:-1].unsqueeze(-3))
    r5 = LogSemiring.dot(r1[...,:-1].unsqueeze(-2),
                         r2.transpose(-2, -1)[...,1:].unsqueeze(-3))


    a = dense_to_sparse(r1, start, offset= 0, semiring=LogSemiring)
    a = bot_pad(a, 1, -2, semiring=LogSemiring)
    a = flip(a, start, semiring=LogSemiring)

    b = dense_to_sparse(r2, start, offset= 0, semiring=LogSemiring)
    b = top_pad(b, 1, -2, semiring=LogSemiring)


    c2 = sparse_combine(b, a,
                        semiring=LogSemiring,
                        fn=lambda a, b: torch.logsumexp(a + b, dim=-1))

    c2 = c2[..., 1:, :]
    c3 = dense_to_sparse(r4, end, offset=-1, semiring=LogSemiring)
    assert torch.isclose(c2.exp(), c3.exp()).all()

    c2 = top_pad(c2, 2, -1, semiring=LogSemiring)
    p = sparse_to_dense(c2, semiring=LogSemiring)
    assert torch.isclose(p.exp(), r4.exp()).all()


    # flip
    a = dense_to_sparse(r1, start, offset= 0, semiring=LogSemiring)
    a = top_pad(a, 1, -2, semiring=LogSemiring)
    a = flip(a, start, semiring=LogSemiring)

    b = dense_to_sparse(r2, start, offset= 0, semiring=LogSemiring)
    b = bot_pad(b, 1, -2, semiring=LogSemiring)


    c2 = sparse_combine(b, a,
                        semiring=LogSemiring,
                        fn=lambda a, b: torch.logsumexp(a + b, dim=-1))

    c2 = c2[..., :-1, :]
    print(c2.shape)
    c3 = dense_to_sparse(r5, end, offset=1, semiring=LogSemiring)
    assert torch.isclose(c2.exp(), c3.exp()).all()

    p = sparse_to_dense(c2, offset=1, semiring=LogSemiring)
    assert torch.isclose(p.exp(), r5.exp()).all()

    c2 = bot_pad(c2, 2, -1, semiring=LogSemiring)
    p = sparse_to_dense(c2, semiring=LogSemiring)
    assert torch.isclose(p.exp(), r5.exp()).all()
