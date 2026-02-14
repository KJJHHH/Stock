import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        # Match input length after padding/dilation so residual add is valid.
        input_len = x.size(-1)
        out = self.net(x)
        if out.size(-1) > input_len:
            out = out[:, :, -input_len:]
        if self.downsample is not None:
            x = self.downsample(x)
        return out + x


class TCNModel(nn.Module):
    def __init__(
        self,
        input_dim=1,
        channels=(64, 64, 64),
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()
        layers = []
        in_channels = input_dim
        for i, out_channels in enumerate(channels):
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x[:, :, -1]
        return self.head(x)
