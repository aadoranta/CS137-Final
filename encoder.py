import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, num_hiddens1, num_hiddens2):
        super(Encoder, self).__init__()

        self.layer1 = nn.Linear(input_dim, num_hiddens1)

        self.layer_norm1 = nn.LayerNorm(num_hiddens1)

        self.layer2 = nn.Linear(num_hiddens1, num_hiddens2)

        self.layer_norm2 = nn.LayerNorm(num_hiddens2)


    def forward(self, inputs):

        x = self.layer1(inputs)
        x = F.relu(x)
        x = self.layer_norm1(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer_norm2(x)

        return x
