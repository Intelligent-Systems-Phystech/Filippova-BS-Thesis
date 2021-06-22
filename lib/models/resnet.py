from torch import nn

from lib.models.base import Backbone

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(Backbone):
    def __init__(self, channels, dropout_rate, return_embed=False, need_softmax=False):
        super().__init__()
        self.channels = channels
        self.dropout_rate = dropout_rate
        self.need_softmax = need_softmax
        self.return_embed = return_embed

        self.conv1 = nn.Conv1d(channels[0], channels[1], kernel_size=7, stride=2, padding=3, bias=True)
        self.bn = nn.BatchNorm1d(channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.downsample0 = nn.Sequential(
            nn.Conv1d(channels[1], channels[2], kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(channels[2])
        )
        
        self.layer1 = nn.Sequential(
            ResidualBlock(channels[1], channels[2], stride=2, downsample=self.downsample0),
            ResidualBlock(channels[2], channels[2])
        )

        self.downsample1 = nn.Sequential(
            nn.Conv1d(channels[2], channels[3], kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(channels[3])
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(channels[2], channels[3], stride=2, downsample=self.downsample1),
            ResidualBlock(channels[3], channels[3])
        )

        self.downsample2 = nn.Sequential(
            nn.Conv1d(channels[3], channels[4], kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(channels[4])
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(channels[3], channels[4], stride=2, downsample=self.downsample2),
            ResidualBlock(channels[4], channels[4])
        )

        self.downsample3 = nn.Sequential(
            nn.Conv1d(channels[4], channels[5], kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(channels[5])
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(channels[4], channels[5], stride=2, downsample=self.downsample3),
            ResidualBlock(channels[5], channels[5])
        )

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(channels[5], channels[6])

        if self.need_softmax:
            self.softmax_out = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        embed = out.view(out.size(0), -1)
        out = self.dropout(embed)
        out = self.fc(out)

        if self.need_softmax:
            out = self.softmax_out(out)

        if self.return_embed:
            return embed, out
        else:
            return out
