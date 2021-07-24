import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 64, 3, padding=1)
        self.maxp1 = torch.nn.MaxPool1d(2, padding=0)
        self.conv2 = torch.nn.Conv1d(64, 128, 3, padding=1)
        self.maxp2 = torch.nn.MaxPool1d(2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxp2(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upsa1 = torch.nn.Upsample(8000)
        self.conv3 = torch.nn.Conv1d(128, 64, 3, padding=1)
        self.upsa2 = torch.nn.Upsample(16000)
        self.conv4 = torch.nn.Conv1d(64, 1, 3, padding=1)

    def forward(self, x):
        x = self.upsa1(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.upsa2(x)
        x = self.conv4(x)
        x = torch.tanh(x)

        return x


class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
