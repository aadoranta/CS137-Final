import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens1, num_hiddens2):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=1,
                                 kernel_size=3,
                                 stride=1, padding='same')

        self.layer1 = nn.Linear(in_channels, num_hiddens1)

        self.layer_norm1 = nn.LayerNorm(num_hiddens1)

        self.layer2 = nn.Linear(num_hiddens1, num_hiddens2)

        self.layer_norm2 = nn.LayerNorm(num_hiddens2)

        self.relu = nn.ReLU()


    def forward(self, inputs):
        # x = self._conv_1(inputs)
        # x = F.relu(x)
        # x = torch.squeeze(x, dim=1)
        x = torch.flatten(inputs)
        x = torch.unsqueeze(x, dim=0)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer_norm1(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer_norm2(x)

        return x

