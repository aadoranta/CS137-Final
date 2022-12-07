import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding='same')

        self._conv_2 = nn.Conv1d(in_channels=num_hiddens,
                                 out_channels=1,
                                 kernel_size=3,
                                 stride=1, padding='same')

    def forward(self, inputs):

        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)

        return x
