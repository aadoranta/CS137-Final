import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=2,
                                 stride=1, padding='same',
                                 )

    def forward(self, inputs):

        x = self._conv_1(inputs)
        x = F.relu(x)

        return x
