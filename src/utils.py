from typing import NoReturn, Literal, Tuple
import torch
from dataclasses import dataclass


class Utils:
    @staticmethod
    def get_separable_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        dilation: int,
        bias: bool = True,
        padding_mode: Literal["zeros", "same", "valid"] = "zeros",
    ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Build separable conv by concating two convs"""
        return [
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                stride=stride,
                dilation=dilation,
                padding=padding,
                kernel_size=kernel_size,
                groups=in_channels,
                bias=bias,
                padding_mode=padding_mode,
            ),
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                dilation=1,
                bias=bias,
                padding_mode=padding_mode,
            ),
        ]

    @staticmethod
    def get_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        dilation: int,
        bias: bool = True,
        padding_mode: Literal["zeros", "same", "valid"] = "zeros",
    ) -> torch.nn.Module:
        """Build regular conv"""
        return torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
        )

    @staticmethod
    def build_conv_block(
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilation: int = 1,
        dropout: float = 0,
        separable: bool = True,
        norm: bool = True,
        activation: bool = True,
    ) -> None:
        """Build one convolutional block for QuartzNet"""
        block = []
        derived_padding = int(dilation * (kernel_size - 1) / 2)
        if separable:
            block.extend(
                Utils.get_separable_conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    derived_padding,
                    stride,
                    dilation,
                )
            )
        else:
            block.append(
                Utils.get_conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    derived_padding,
                    stride,
                    dilation,
                )
            )

        if norm:
            block.append(torch.nn.BatchNorm1d(out_channels))
        if activation:
            block.append(torch.nn.ReLU())
            block.append(torch.nn.Dropout(dropout))

        return block

    @staticmethod
    def build_block_sequence(
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilation: int,
        separable: bool,
        dropout: float,
        repeat: bool,
    ) -> torch.nn.Module:
        output = torch.nn.ModuleList()
        if repeat <= 1:
            output.extend(
                Utils.build_conv_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    activation=True,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    separable=separable,
                    norm=True,
                    dropout=dropout,
                )
            )
            repeat = 0

        for i in range(repeat):
            if not i:
                output.extend(
                    Utils.build_conv_block(
                        in_channels=in_channels,
                        activation=True,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        separable=separable,
                        out_channels=out_channels,
                        norm=True,
                        dropout=dropout,
                    )
                )
            elif i == repeat - 1:
                output.extend(
                    Utils.build_conv_block(
                        in_channels=out_channels,
                        activation=False,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        separable=separable,
                        out_channels=out_channels,
                        norm=True,
                        dropout=dropout,
                    )
                )
            else:
                output.extend(
                    Utils.build_conv_block(
                        in_channels=out_channels,
                        activation=True,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        separable=separable,
                        out_channels=out_channels,
                        norm=True,
                        dropout=dropout,
                    )
                )
        return output
