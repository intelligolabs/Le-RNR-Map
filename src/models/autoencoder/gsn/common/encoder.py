import math
import torch
import torch.nn as nn
from .layers import ConvLayer2d, ConvResBlock2d, EqualLinear


class DiscriminatorHead(nn.Module):
    def __init__(self, in_channel, disc_stddev=False):
        super().__init__()

        self.disc_stddev = disc_stddev
        stddev_dim = 1 if disc_stddev else 0

        self.conv_stddev = ConvLayer2d(
            in_channel=in_channel + stddev_dim, out_channel=in_channel, kernel_size=3, activate=True
        )

        self.final_linear = nn.Sequential(
            nn.Flatten(),
            EqualLinear(in_channel=in_channel * 4 * 4, out_channel=in_channel, activate=True),
            EqualLinear(in_channel=in_channel, out_channel=1),
        )

    def cat_stddev(self, x, stddev_group=4, stddev_feat=1):
        perm = torch.randperm(len(x))
        inv_perm = torch.argsort(perm)

        batch, channel, height, width = x.shape
        x = x[perm]  # shuffle inputs so that all views in a single trajectory don't get put together

        group = min(batch, stddev_group)
        stddev = x.view(group, -1, stddev_feat, channel // stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)

        stddev = stddev[inv_perm]  # reorder inputs
        x = x[inv_perm]

        out = torch.cat([x, stddev], 1)
        return out

    def forward(self, x):
        if self.disc_stddev:
            x = self.cat_stddev(x)
        x = self.conv_stddev(x)
        out = self.final_linear(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self, in_channel, out_channel, in_res, out_res):
        super().__init__()

        log_size_in = int(math.log(in_res, 2))
        log_size_out = int(math.log(out_res, 2))

        self.layers = []
        in_ch = in_channel
        for i in range(log_size_in, log_size_out):
            out_ch = in_ch // 2
            self.layers.append(
                ConvLayer2d(
                    in_channel=in_ch, out_channel=out_ch, kernel_size=3, upsample=True, bias=True, activate=True
                )
            )
            in_ch = out_ch

        self.layers.append(
            ConvLayer2d(in_channel=in_ch, out_channel=out_channel, kernel_size=3, bias=True, activate=False)
        )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, in_channel, in_res, out_res, ch_mul=64, ch_max=32, lasts_ch=32, **kwargs):
        super().__init__()

        log_size_in = int(math.log(in_res, 2))
        log_size_out = int(math.log(out_res, 2))

        self.conv_in = ConvLayer2d(in_channel=in_channel, out_channel=ch_mul, kernel_size=3)

        # each resblock will half the resolution and double the number of features (until a maximum of ch_max)
        self.layers = []
        in_channels = ch_mul
        for i in range(log_size_in, log_size_out, -1):
            out_channels = int(min(in_channels * 2, ch_max))
            self.layers.append(ConvResBlock2d(in_channel=in_channels, out_channel=out_channels, downsample=True))
            in_channels = out_channels

        self.layers = nn.Sequential(*self.layers)
        self.conv_out = ConvLayer2d(in_channel=in_channels, out_channel=lasts_ch, kernel_size=1, activate=False)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.layers(x)
        x = self.conv_out(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)