import torch

def LogMemSemiring(max_size=100000):
    class _LogMemDot(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            size = [max(p, q) for p, q in zip(a.shape, b.shape)][:-1]
            ret = torch.zeros(*size, dtype=a.dtype, device=a.device)
            accumulate_(a, b, ret,
                        lambda a, b: torch.logsumexp(a + b, dim=-1),
                        preserve=len(ret.shape),
                        step=max_size // a.shape[-1] + 2)
            return ret

        @staticmethod
        def backward(ctx, grad_output):
            a, b = ctx.saved_tensors

            size = [max(p, q) for p, q in zip(a.shape, b.shape)][:-1]

            fn = lambda a, b, g: torch.softmax(a + b, dim=-1).mul(g.unsqueeze(-1))
            if True:
                grad_a, grad_b = unaccumulate_(
                    a, b, grad_output, fn,
                    step=max_size // a.shape[-1] + 2
                )
            else:
                asum, bsum = [], []
                for i, (x, y) in enumerate(zip(a.shape, b.shape)):
                    if x == 1:
                        asum.append(i)
                    if y == 1:
                        bsum.append(i)
                back = fn(a, b, grad_output)
                grad_a = back.sum(dim=asum, keepdim=True)
                grad_b = back.sum(dim=bsum, keepdim=True)

            return grad_a, grad_b

    class _LogMemBandedDot2(torch.autograd.Function):
        @staticmethod
        def compute(a, b, band, o1, o2):
            if False:
                return run(a, b)
            else:
                x = a.shape[-2]
                size = a.shape[-1]
                y = (size - 1) * 2 + 3
                size = [max(p, q) for p, q in zip(a.shape, b.shape)][:-2]
                size.append(x)
                size.append(y)
                ret = torch.zeros(*size, dtype=a.dtype, device=a.device)
                accumulate_(a, b, ret, preserve=-2,
                            fn=_LogMemBandedDot2.run(band, o1, o2),
                            step=max_size // (a.shape[-2] * a.shape[-1]) + 2)
                return ret

        @staticmethod
        def run(band, o1, o2):
            def _run(a, b):
                return sparse_banded_combine2(a, b, band, o1, o2,
                                              semiring=LogSemiring,
                                              fn=lambda a, b: torch.logsumexp(a + b, dim=-1))
            return _run

        @staticmethod
        def forward(ctx, a, b, band, o1, o2):
            ctx.save_for_backward(a, b, torch.tensor([band, o1, o2]))
            return _LogMemBandedDot2.compute(a, b, band, o1, o2)

        @staticmethod
        def backward(ctx, grad_output):
            a, b, opt = ctx.saved_tensors
            band, o1, o2 = opt.tolist()


            def run(a, b):
                return sparse_banded_combine2(
                    a, b, band, o1, o2,
                    semiring=LogSemiring,
                    fn=lambda a, b: torch.logsumexp(a + b, dim=-1))
            grad_a, grad_b = unaccumulate2_(a, b, grad_output,
                                            preserve=-2, fn=run,
                                            step=max_size // (a.shape[-2] * a.shape[-1]) + 2)
            return grad_a, grad_b, None, None, None


    class _LogMemSemiring(_BaseLog):
        """
        Implements the log-space semiring (logsumexp, +, -inf, 0).

        Gradients give marginals.
        """

        @staticmethod
        def sum(xs, dim=-1):
            return torch.logsumexp(xs, dim=dim)

        @classmethod
        def dot(cls, a, b):
            "Dot product along last dim."
            return _LogMemDot.apply(a, b, )

        @classmethod
        def banded_dot2(cls, a, b, band, offset_a, offset_b):
            return _LogMemBandedDot2.apply(a, b, band, offset_a, offset_b)

        @classmethod
        def dot_grad(cls, a, b):
            "Dot product along last dim."
            c = a + b
            part = torch.logsumexp(c, dim=-1)
            return part, (c - part.unsqueeze(-1)).exp()
    return _LogMemSemiring


def ones(x):
    one = []
    for i, v in enumerate(x.shape[:-1]):
        if v == 1:
            one.append(i)
    return one

def mind(one, inds):
    inds = list(inds)
    for v in one:
        inds[v] = inds[v].clone().fill_(0)
    return inds

def accumulate_(a, b, ret, fn, preserve, step=10000):
    slices = []
    total = 1
    for s in ret.shape[:preserve]:
        slices.append(slice(s))
        total *= s

    a_one, b_one = ones(a), ones(b)
    indices = torch.tensor(np.mgrid[slices]).view(len(ret.shape[:preserve]), -1)

    for p in range(0, total, step):
        ind = indices[:, p : p + step].unbind()
        a_ind = mind(a_one, ind)
        b_ind = mind(b_one, ind)
        ret[ind] = fn(a[tuple(a_ind)], b[tuple(b_ind)])


def unaccumulate_(a, b, grad_output, fn, step=10000):
    slices = []
    a_grad = a.clone().fill_(0)
    b_grad = b.clone().fill_(0)

    total = 1
    for s in grad_output.shape:
        slices.append(slice(s))
        total *= s
    a_one, b_one = ones(a), ones(b)

    indices = torch.tensor(np.mgrid[slices]).view(len(grad_output.shape), -1)

    for p in range(0, total, step):
        ind = indices[:, p : p + step].unbind()
        a_ind = mind(a_one, ind)
        b_ind = mind(b_one, ind)

        q = fn(a[tuple(a_ind)], b[tuple(b_ind)], grad_output[tuple(ind)])
        a_grad.index_put_(tuple(a_ind),  q, accumulate=True)
        b_grad.index_put_(tuple(b_ind),  q, accumulate=True)
    return a_grad, b_grad

def unaccumulate2_(a, b, grad_output, preserve, fn, step=10000):
    slices = []
    a_grad = a.clone().fill_(0)
    b_grad = b.clone().fill_(0)

    total = 1
    for s in grad_output.shape[:preserve]:
        slices.append(slice(s))
        total *= s
    a_one, b_one = ones(a), ones(b)

    indices = torch.tensor(np.mgrid[slices]).view(len(grad_output.shape[:preserve]), -1)

    for p in range(0, total, step):
        ind = indices[:, p : p + step].unbind()
        a_ind = mind(a_one, ind)
        b_ind = mind(b_one, ind)

        with torch.enable_grad():
            a_in = a.clone().requires_grad_(True)
            b_in = b.clone().requires_grad_(True)
            q = fn(a[tuple(a_ind)], b[tuple(b_ind)])
        ag, bg = torch.autograd.grad(q, (a, b), grad_output[tuple(ind)])
        a_grad += ag
        b_grad += bg

    return a_grad, b_grad


    # class _LogMemBandedDot(torch.autograd.Function):
    #     @staticmethod
    #     def forward(ctx, a, b, band, o1, o2):
    #         ctx.save_for_backward(a, b, torch.tensor([band, o1, o2]))
    #         if True:
    #             return sparse_banded_combine2(a, b, band, o1, o2,
    #                                           semiring=LogSemiring,
    #                                           fn=lambda a, b: torch.logsumexp(a + b, dim=-1))
    #         else:
    #             return sparse_banded_combine(a, b, band, o1, o2,
    #                                          semiring=LogSemiring,
    #                                          fn=lambda a, b: torch.logsumexp(a + b, dim=-1))


    #     @staticmethod
    #     def backward(ctx, grad_output):
    #         a, b, opt = ctx.saved_tensors
    #         band, o1, o2 = opt.tolist()
    #         next_band = (band - 1) * 2 + 1

    #         size = [max(p, q) for p, q in zip(a.shape, b.shape)][:-1]
    #         def fn(a, b, gr):
    #             g = dense_to_sparse(gr, next_band, semiring=StdSemiring)
    #             g2 = dense_to_sparse(gr.transpose(-2,-1), next_band,
    #                                  semiring=StdSemiring)
    #             def inner(a, b):
    #                 return torch.softmax(a+b, -1).mul(g.unsqueeze(-1)).sum(-2)
    #             def inner2(a, b):
    #                 return torch.softmax(a+b, -1).mul(g2.unsqueeze(-1)).sum(-2)

    #             grad1, grad2 = sparse_banded_grad(a, b, band, o1, o2,
    #                                               semiring=LogSemiring,
    #                                               fn=inner, fn2=inner2)
    #             return grad1, grad2
    #         if True:
    #             asum, bsum = [], []
    #             for i, (x, y) in enumerate(zip(a.shape, b.shape)):
    #                 if x == 1:
    #                     asum.append(i)
    #                 if y == 1:
    #                     bsum.append(i)
    #             grad_a, grad_b = fn(a, b, grad_output)
    #             grad_a = grad_a.sum(dim=asum, keepdim=True)
    #             grad_b = grad_b.sum(dim=bsum, keepdim=True)
    #         else:
    #             grad_a, grad_b = unaccumulate_(
    #                 a, b, grad_output, fn,
    #                 step=max_size // a.shape[-1] + 2
    #             )

    #         return grad_a, grad_b, None, None, None
