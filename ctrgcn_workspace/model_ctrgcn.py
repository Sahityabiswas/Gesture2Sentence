import torch
import torch.nn as nn

import config
from graph import build_adjacency


class CTRGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.pre = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.theta = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.phi = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.out = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj):
        x_proj = self.pre(x)
        theta = self.theta(x_proj).mean(dim=2)
        phi = self.phi(x_proj).mean(dim=2)
        relation = torch.tanh(theta.unsqueeze(-1) - phi.unsqueeze(-2))
        relation = adj.unsqueeze(0).unsqueeze(0) + self.alpha * relation
        y = torch.einsum("bcuv,bctv->bctu", relation, x_proj)
        return self.out(y)


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=9, stride=1, dropout=0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(pad, 0),
            ),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CTRGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, stride=1, dropout=0.0):
        super().__init__()
        self.gcn = CTRGraphConv(in_channels, out_channels, num_nodes)
        self.tcn = TemporalConv(out_channels, stride=stride, dropout=dropout)
        self.relu = nn.ReLU(inplace=True)

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, adj):
        res = self.residual(x)
        x = self.gcn(x, adj)
        x = self.tcn(x)
        return self.relu(x + res)


class CTRGCNSignModel(nn.Module):
    def __init__(
        self,
        num_classes,
        num_nodes=config.NUM_KEYPOINTS,
        in_channels=config.INPUT_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        dropout=config.DROPOUT,
    ):
        super().__init__()
        adj = build_adjacency(num_nodes)
        self.register_buffer("adjacency", adj)

        self.data_bn = nn.BatchNorm1d(num_nodes * in_channels)
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=1)

        self.layers = nn.ModuleList([
            CTRGCNBlock(base_channels, base_channels, num_nodes, dropout=dropout),
            CTRGCNBlock(base_channels, base_channels, num_nodes, dropout=dropout),
            CTRGCNBlock(base_channels, base_channels * 2, num_nodes, stride=2, dropout=dropout),
            CTRGCNBlock(base_channels * 2, base_channels * 2, num_nodes, dropout=dropout),
        ])

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, num_classes),
        )

    def forward(self, x, lengths=None):
        b, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, v * c, t)
        x = self.data_bn(x)
        x = x.view(b, v, c, t).permute(0, 2, 3, 1).contiguous()
        x = self.stem(x)

        for layer in self.layers:
            x = layer(x, self.adjacency)

        x = x.mean(dim=-1).mean(dim=-1)
        return self.head(x)
