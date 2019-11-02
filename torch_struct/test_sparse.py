from .sparse import *
import torch

def test_sparse():
    r = torch.rand(5, 10, 10)
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
    
