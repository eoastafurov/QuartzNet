from typing import List

import torch
from torch import nn

from src.utils import Utils


class QuartzNetBlock(torch.nn.Module):
    def __init__(
        self,
        feat_in: int,
        filters: int,
        repeat: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        residual: bool,
        separable: bool,
        dropout: float,
    ):

        super().__init__()

        self.res = (
            None
            if not residual
            else torch.nn.Sequential(
                *Utils.build_conv_block(
                    in_channels=feat_in,
                    out_channels=filters,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    separable=False,
                    norm=True,
                    activation=False,
                )
            )
        )

        self.conv = Utils.build_block_sequence(
            kernel_size=kernel_size,
            in_channels=feat_in,
            out_channels=filters,
            stride=stride,
            dilation=dilation,
            separable=separable,
            dropout=dropout,
            repeat=repeat,
        )

        self.out = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual = self.res(x) if self.res else 0
        # for block in self.conv:
        #     x = block(x)
        # x += residual
        # x = self.out(x)
        # return x
        # print(12345567)
        if self.res:
            residual = self.res(x)
        for layer in self.conv:
            x = layer(x)
        if self.res:
            x += residual
        return self.out(x)


class QuartzNet(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.stride_val = 1

        layers = []
        feat_in = conf.feat_in
        for block in conf.blocks:
            # print(block)
            layers.append(QuartzNetBlock(feat_in, **block))
            self.stride_val *= block.stride**block.repeat
            feat_in = block.filters

        self.layers = nn.Sequential(*layers)
        # print(self.layers)

    def forward(
        self, features: torch.Tensor, features_length: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.layers(features)
        encoded_len = (
            torch.div(features_length - 1, self.stride_val, rounding_mode="trunc") + 1
        )

        return encoded, encoded_len
