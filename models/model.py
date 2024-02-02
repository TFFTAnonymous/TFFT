"""
model.py

Implementation of Tailored FFT (T-FFT).
Note that this code example is specifically designed for
a Taylor kernel of characteristic 1 for the best performance.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Spec1d(nn.Module):
    """
    Encoder of Tailored FFT
    """
    def __init__(self, p_order=None, kernel_size=None):
        super().__init__()
        self.L = p_order
        self.K = kernel_size - 1
        self.h = nn.Parameter(torch.zeros(1, dtype=torch.float))
        self.a = nn.Parameter(torch.ones(1, dtype=torch.cfloat))
        self.b = nn.Parameter(torch.randn(1, dtype=torch.cfloat))
        self.W = nn.Parameter((torch.rand(self.K, dtype=torch.cfloat) - 0.5 - 0.5j) / 5.0)

    def forward(self, x):
        # Fourier transform
        out = torch.fft.rfft(x)
        # Phase shifting
        phase = torch.exp(2.0j * torch.pi * self.h * torch.arange(0, out.shape[0]))
        out = out * phase
        # Normalization
        o = out[:1]
        out = out[1:]
        out = self.a * out + self.b
        # Taylor transform
        out = self.taylor_transform(out, self.W)
        return o, self.h, self.a, self.b, self.W, out

    def taylor_transform(self, x, W):
        """
        Compute the one dimensional Taylor transform of x
        :param x: Input tensor of shape (M, )
        :param W: Taylor kernel of shape (K, )
        :return:  Taylor coefficients of shape (K, L)
        """
        if self.K == 1:
            res = [x[0]]
            for _ in range(self.L - 1):
                x = x[1:] + x[:-1] * W
                res.append(x[0])
            return torch.tensor(res).view(1, -1)
        else:
            res = torch.empty(0, dtype=torch.cfloat)
            W = torch.cat((W, torch.ones(1))).view(1, 1, -1)
            x = x.view(1, 1, -1)
            for _ in range(self.L - 1):
                res = torch.cat([res, x[:, :, :self.K]])
                x = F.conv1d(x, W)
            if x.shape[-1] < self.K:
                pad = torch.zeros(1, 1, self.K - x.shape[-1])
                res = torch.cat((res, torch.cat((x, pad))))
            else:
                res = torch.cat([res, x[:, :, :self.K]])
            return res.view(self.L, self.K).T


class SpecInverse1d(nn.Module):
    """
    Decoder of Tailored FFT
    """
    def __init__(self, out_size=None):
        super().__init__()
        self.N = out_size
        self.o = None
        self.h = None
        self.a = None
        self.b = None
        self.W = None
        self.K = None

    def forward(self, config):
        self.o = config[0]
        self.h = config[1]
        self.a = config[2]
        self.b = config[3]
        self.W = config[4]
        self.K = self.W.shape[-1]
        out = config[-1]
        # Inverse Taylor transform
        out = self.inverse_taylor_transform(out, self.W, self.N // 2)
        # Denormalization
        out = (out - self.b) / self.a
        out = torch.cat((self.o, out))
        # Phase shifting
        phase = torch.exp(-2.0j * torch.pi * self.h * torch.arange(0, out.shape[0]))
        out = out * phase
        # Inverse Fourier transform
        out = torch.fft.irfft(out, self.N)
        return out

    def inverse_taylor_transform(self, x, W, M):
        """
        Compute the one dimensional inverse Taylor transform of x
        :param x: Input tensor of shape (K, L)
        :param W: Taylor kernel of shape (K, )
        :param M: Output shape descriptor
        :return:  Output tensor of shape (M, )
        """
        L = x.shape[-1]
        P = self.d_polynomial((M, L), W)
        x = torch.matmul(P, x.T.ravel())
        return x

    def d_polynomial(self, shape, W):
        """
        Compute d-polynomials
        :param shape: Shape of each d-polynomial (M, L)
        :param W: Taylor kernel of shape (K, )
        :return:  d-polynomial batch of shape (M, K * L)
        """
        if self.K == 1:
            P = torch.zeros(shape[1], dtype=torch.cfloat)
            P[0] = 1.0
            res = [P]
            for _ in range(shape[0] - 1):
                P = F.pad(P[:-1], pad=(1, 0)) - P * W
                res.append(P)
            return torch.stack(res)
        else:
            P = torch.zeros((self.K, shape[1] * self.K), dtype=torch.cfloat)
            V = torch.tensor([1.0, -W[0]], dtype=torch.cfloat).view(1, 1, -1)
            for k in range(self.K):
                P[k, shape[1] * k] = 1.0
            for k in range(self.K, shape[0]):
                temp = F.conv1d(P[k - self.K].view(1, 1, -1), V, padding=1).ravel()[:P.shape[1]]
                for j in range(1, self.K):
                    temp -= self.W[-j] * P[k - j]
                temp = temp.view(1, -1)
                P = torch.cat((P, temp))
            return P


class TFFT(nn.Module):
    """
    Tailored FFT
    """
    def __init__(self, in_size=None, hidden_size=None):
        super(TFFT, self).__init__()
        self.encoder = Spec1d(p_order=hidden_size, kernel_size=2)
        self.decoder = SpecInverse1d(out_size=in_size)

    def forward(self, x):
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out)
        return enc_out, dec_out


class DECODE(nn.Module):
    """
    Decoder of Tailored FFT for reconstruction
    """
    def __init__(self, out_size=None):
        super(DECODE, self).__init__()
        self.decoder = SpecInverse1d(out_size=out_size)

    def forward(self, encoded):
        return self.decoder(encoded)
