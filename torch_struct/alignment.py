import torch
from .helpers import _Struct
import math
from .sparse import *
from pytorch_memlab import MemReporter

class Alignment(_Struct):
    def _check_potentials(self, edge, lengths=None):
        batch, N_1, M_1, x = edge.shape
        assert x == 3
        edge = self.semiring.convert(edge)

        N = N_1
        M = M_1
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)

        assert max(lengths) <= N, "Length longer than edge scores"
        assert max(lengths) == N, "One length must be at least N"
        return edge, batch, N, M, lengths

    def _dp(self, log_potentials, lengths=None, force_grad=False):
        return self._dp_scan(log_potentials, lengths, force_grad)

    def _dp_scan(self, log_potentials, lengths=None, force_grad=False):
        "Compute forward pass by linear scan"
        # Setup
        semiring = self.semiring
        log_potentials.requires_grad_(True)
        ssize = semiring.size()
        log_potentials, batch, N, M, lengths = self._check_potentials(
            log_potentials, lengths
        )
        steps = N + M
        log_MN = int(math.ceil(math.log(steps, 2)))
        bin_MN = int(math.pow(2, log_MN))

        Down, Mid, Up = 0, 1, 2
        Open, Close = 0, 1
        # Create a chart N, N, back

                  
        chart = [None for i in range(log_MN + 1)]
        charta = [None for i in range(log_MN + 1)]
        chartb = [None for i in range(log_MN + 1)]
        charta[0] = self._make_chart(
            1, (batch, bin_MN, 1, bin_MN,  2, 2, 3), log_potentials, force_grad
        )[0]
        chartb[0] = self._make_chart(
            1, (batch, bin_MN, bin_MN, 1,  2, 2, 3), log_potentials, force_grad
        )[0]

        charta[1] = self._make_chart(
            1, (batch, bin_MN // 2, 3, bin_MN, 2, 2, 3), log_potentials, force_grad
        )[0]
        
        # Init
        # This part is complicated. Rotate the scores by 45% and
        # then compress one.
        grid_x = torch.arange(N).view(N, 1).expand(N, M)
        grid_y = torch.arange(M).view(1, M).expand(N, M)
        rot_x = grid_x + grid_y
        rot_y = grid_y - grid_x + N
        ind = torch.arange(bin_MN)
        ind_M = ind
        ind_U = torch.arange(1, bin_MN)
        ind_D = torch.arange(bin_MN - 1)

        for b in range(lengths.shape[0]):
            end = lengths[b]
            point = (end + M) // 2
            lim = point * 2
            
            charta[0][:, b, rot_x[:lim], 0, rot_y[:lim], :, :, :] = (
                log_potentials[:, b, :lim].unsqueeze(-2).unsqueeze(-2)
            )
            chartb[0][:, b, rot_x[:lim], rot_y[:lim], 0, :, :, :] = (
                log_potentials[:, b, :lim].unsqueeze(-2).unsqueeze(-2)
            )
            
            charta[1][:, b, point:, 1, ind, :, :, Mid] = semiring.one_(
                charta[1][:, b, point:, 1, ind, :, :, Mid]
            )
            
        for b in range(lengths.shape[0]):
            end = lengths[b]
            point = (end + M) // 2
            lim = point * 2
            
            left2_ = charta[0][:, b, 0:lim:2]
            right2 = chartb[0][:, b, 1:lim:2]
            
            charta[1][:, b, :point, 1, ind_M, :, :, :] = torch.stack(
                [
                    left2_[:, :, 0, ind_M, :, :, Down],
                    semiring.plus(
                        left2_[:, :, 0, ind_M, :, :, Mid],
                        right2[:, :, ind_M, 0, :, :, Mid],
                    ),
                    left2_[:, :, 0, ind_M, :, :, Up],
                ],
                dim=-1,
            )
            
            y = torch.stack([ind_D, ind_U], dim=0)
            z = y.clone()
            z[0, :] = 2
            z[1, :] = 0
            
            z2 = y.clone()
            z2[0, :] = 0
            z2[1, :] = 2
        
            tmp = torch.stack(
                [
                    semiring.times(
                        left2_[:, :, 0, ind_D, Open : Open + 1 :, :],
                        right2[:, :, ind_U, 0, :, Open : Open + 1, Down : Down + 1],
                    ),
                    semiring.times(
                        left2_[:, :, 0, ind_U, Open : Open + 1, :, :],
                        right2[:, :, ind_D, 0, :, Open : Open + 1, Up : Up + 1],
                    ),
                ],
                dim=2,
            )
            charta[1][:, b, :point, z, y, :, :, :] = tmp

        
        # for b in range(lengths.shape[0]):
        #     end = lengths[b]
        #     # Add path to end.
        #     point = (end + M) // 2
        #     point = (end + M) // 2
        #     lim = point * 2
        #     chart[1][:, b, point : bin_MN // 2, ind, ind, Mid] = semiring.one_(
        #         chart[1][:, b, point : bin_MN // 2, ind, ind, Mid]
        #     )
        #     chart[0][
        #         :, b, rot_x[: end + M], rot_y[:lim], rot_y[:lim], :
        #     ] = log_potentials[:, b, : end + M]

        # for b in range(lengths.shape[0]):
        #     end = lengths[b]
        #     point = (end + M) // 2
        #     lim = point * 2
        #     chart[1][:, b, :point, ind_M, ind_M, :] = torch.stack(
        #         [
        #             chart[0][:, b, :lim:2, ind_M, ind_M, Down],
        #             semiring.sum(
        #                 torch.stack(
        #                     [
        #                         chart[0][:, b, :lim:2, ind_M, ind_M, Mid],
        #                         chart[0][:, b, 1:lim:2, ind_M, ind_M, Mid],
        #                     ],
        #                     dim=-1,
        #                 )
        #             ),
        #             chart[0][:, b, :lim:2, ind_M, ind_M, Up],
        #         ],
        #         dim=-1,
        #     )

        #     x = torch.stack([ind_U, ind_D], dim=0)
        #     y = torch.stack([ind_D, ind_U], dim=0)
        #     q = torch.stack(
        #         [
        #             semiring.times(
        #                 chart[0][:, b, :lim:2, ind_D, ind_D, :],
        #                 chart[0][:, b, 1:lim:2, ind_U, ind_U, Down : Down + 1],
        #             ),
        #             semiring.times(
        #                 chart[0][:, b, :lim:2, ind_U, ind_U, :],
        #                 chart[0][:, b, 1:lim:2, ind_D, ind_D, Up : Up + 1],
        #             ),
        #         ],
        #         dim=2,
        #     )

        #     chart[1][:, b, :point, x, y, :] = q
        
        size = bin_MN // 2
        c = charta[1][..., 0, 0, :].view(ssize, batch, size, 3, bin_MN, 3) \
                    .permute(0, 1, 2, 5, 4, 3)  \
                    .view(ssize, batch, size, 3, bin_MN, 3)
        c2 = sparse_to_dense(c, semiring=semiring)
        chart[1] = c2.view(ssize, batch, size, 3, bin_MN, bin_MN) \
                   .permute(0, 1, 2, 4, 5, 3)
        # print(chart[1])
        # print(charta[1])
        # Scan
        def merge(x, size, rsize):
            left = (
                x[:, :, 0 : size * 2 : 2]
                .permute(0, 1, 2, 5, 4, 3)
                .view(ssize, batch, size, 3, bin_MN,  bin_MN)
            )
            right = (
                x[:, :, 1 : size * 2 : 2]
                .permute(0, 1, 2,  5, 3, 4)
                .view(ssize, batch, size, 1, 3,  bin_MN, bin_MN)
            )
            st = []
            for op in (Up, Down, Mid):
                a, b, c, d = 0, bin_MN, 0, bin_MN
                if op == Up:
                    a, b, c, d = 1, bin_MN, 0, bin_MN - 1
                if op == Down:
                    a, b, c, d = 0, bin_MN - 1, 1, bin_MN

                if rsize > 100: 
                    v = semiring.dot(left[..., a:b].unsqueeze(-2),
                                     right[..., op, :,  c:d].unsqueeze(-3))

                else:
                    v = semiring.banded_dot(
                        left[..., :],
                        right[..., op,  :, :].transpose(-2, -1),
                        rsize, c, a)

                v = v.view(ssize, batch, size, 3, bin_MN, bin_MN) \
                    .permute(0, 1, 2, 5, 4, 3) 
                
                st.append(v)
            st = torch.stack(st, dim=-1)
            return semiring.sum(st)
        
        rsize = 2
        size = bin_MN // 2
        for n in range(2, log_MN + 1):
            size = int(size / 2)
            rsize = rsize * 2
            chart[n] = merge(chart[n - 1], size, rsize+1)
        # reporter = MemReporter()
        # reporter.report()

        v = chart[-1][:, :, 0, M, N, Mid]
        return v, [log_potentials], None

    @staticmethod
    def _rand(min_n=2):
        b = torch.randint(2, 4, (1,))
        N = torch.randint(min_n, 4, (1,))
        M = torch.randint(min_n, 4, (1,))
        return torch.rand(b, N, M, 3), (b.item(), (N).item())

    def enumerate(self, edge, lengths=None):
        semiring = self.semiring
        edge, batch, N, M, lengths = self._check_potentials(edge, lengths)
        d = {}
        d[0, 0] = [([(0, 0)], edge[:, :, 0, 0, 1])]
        # enum_lengths = torch.LongTensor(lengths.shape)
        for i in range(N):
            for j in range(M):
                d.setdefault((i + 1, j + 1), [])
                d.setdefault((i, j + 1), [])
                d.setdefault((i + 1, j), [])
                for chain, score in d[i, j]:
                    if i + 1 < N and j + 1 < M:
                        d[i + 1, j + 1].append(
                            (
                                chain + [(i + 1, j + 1)],
                                semiring.mul(score, edge[:, :, i + 1, j + 1, 1]),
                            )
                        )
                    if i + 1 < N:

                        d[i + 1, j].append(
                            (
                                chain + [(i + 1, j)],
                                semiring.mul(score, edge[:, :, i + 1, j, 2]),
                            )
                        )
                    if j + 1 < M:
                        d[i, j + 1].append(
                            (
                                chain + [(i, j + 1)],
                                semiring.mul(score, edge[:, :, i, j + 1, 0]),
                            )
                        )
        all_val = torch.stack([x[1] for x in d[N - 1, M - 1]], dim=-1)
        return semiring.unconvert(semiring.sum(all_val)), None
