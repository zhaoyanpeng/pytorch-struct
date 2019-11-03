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

    # FINISH
    
    # flippeda = pad(b.flip(1), 4, 0).unfold(0, 5, 1).diagonal(0, 1, 2)
    # flippedb = pad(b.flip(1), 4, 0).unfold(0, 5, 1).diagonal(0, 1, 2)
    
    # combo = sparse_combine(a, b, back=True)
    # print(combo)
    # flipped = pad(combo.flip(1), 8, 0).unfold(0, 9, 1).diagonal(0, 1, 2)

    
    # assert torch.isclose(sparse_to_dense(flipped),
    #                      torch.matmul(r1, r2)).all()
    
    
