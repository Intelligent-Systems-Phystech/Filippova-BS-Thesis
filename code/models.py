from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
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


class ResNet(nn.Module):
    def __init__(self, channels):
        super(ResNet, self).__init__()
        self.init_params = channels
        self.conv1 = nn.Conv1d(channels[0], channels[1], kernel_size=7,
                               stride=2, padding=3, bias=True)
        self.bn = nn.BatchNorm1d(channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = nn.Sequential(
            ResidualBlock(channels[1], channels[2]),
            ResidualBlock(channels[2], channels[2]))

        self.downsample1 = nn.Sequential(
            nn.Conv1d(channels[2], channels[3], kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(channels[3]))

        self.layer2 = nn.Sequential(
            ResidualBlock(channels[2], channels[3], stride=2, downsample=self.downsample1),
            ResidualBlock(channels[3], channels[3]))

        self.downsample2 = nn.Sequential(
            nn.Conv1d(channels[3], channels[4], kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(channels[4]))

        self.layer3 = nn.Sequential(
            ResidualBlock(channels[3], channels[4], stride=2, downsample=self.downsample2),
            ResidualBlock(channels[4], channels[4]))

        self.downsample3 = nn.Sequential(
            nn.Conv1d(channels[4], channels[5], kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(channels[5]))

        self.layer4 = nn.Sequential(
            ResidualBlock(channels[4], channels[5], stride=2, downsample=self.downsample3),
            ResidualBlock(channels[5], channels[5]))

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.dropout = nn.Dropout(channels[6])
        self.fc = nn.Linear(channels[5], 2)

    def get_params(self):
        params = {"init_params": self.init_params,
                  "class_name": "ResNet"}

        return params

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
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class LSTM(nn.Module):
    def __init__(self, hyperparameters):
        super(LSTM, self).__init__()
        self.init_params = hyperparameters
        self.input_dim = hyperparameters[0]
        self.hidden_dim = hyperparameters[1]
        self.num_layers = hyperparameters[2]
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        self.linear = nn.Linear(self.hidden_dim, 2)
        self.dropout = nn.Dropout(hyperparameters[3])

    def get_params(self):
        params = {"init_params": self.init_params,
                  "class_name": "LSTM"}

        return params

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x.view(x.shape[0], x.shape[2], -1))
        lstm_out = lstm_out.contiguous()[:, -1].view(x.shape[0], -1)
        out = lstm_out.view(lstm_out.size(0), -1)
        out = self.dropout(out)
        y_pred = self.linear(out)
        return y_pred


class ResNetLSTM(nn.Module):
    def __init__(self, channels):
        super(ResNetLSTM, self).__init__()
        self.init_params = channels
        self.conv1 = nn.Conv1d(channels[0], channels[1], kernel_size=7,
                               stride=2, padding=3, bias=True)
        self.bn = nn.BatchNorm1d(channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = nn.Sequential(
            ResidualBlock(channels[1], channels[2]),
            ResidualBlock(channels[2], channels[2]))

        self.downsample1 = nn.Sequential(
            nn.Conv1d(channels[2], channels[3], kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(channels[3]))

        self.layer2 = nn.Sequential(
            ResidualBlock(channels[2], channels[3], stride=2, downsample=self.downsample1),
            ResidualBlock(channels[3], channels[3]))

        self.downsample2 = nn.Sequential(
            nn.Conv1d(channels[3], channels[4], kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(channels[4]))

        self.layer3 = nn.Sequential(
            ResidualBlock(channels[3], channels[4], stride=2, downsample=self.downsample2),
            ResidualBlock(channels[4], channels[4]))

        self.input_dim = channels[4]
        self.hidden_dim = channels[5]
        self.num_layers = channels[6]
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, 2)
        self.dropout = nn.Dropout(channels[7])

    def get_params(self):
        params = {"init_params": self.init_params,
                  "class_name": "ResNetLSTM"}

        return params

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        lstm_out, self.hidden = self.lstm(out.view(out.shape[0], out.shape[2], -1))
        lstm_out = lstm_out.contiguous()[:, -1].view(out.shape[0], -1)
        out = lstm_out.view(lstm_out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out
