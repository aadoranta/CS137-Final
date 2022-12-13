import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim_2, output_dim):
        super(FeedForward, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        # Non-linearity
        self.relu = nn.Tanh()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim_2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim_2)

        # Linear function (readout)
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.relu(out)

        out = self.layer_norm1(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)

        out = self.relu(out)

        out = self.layer_norm2(out)

        out = self.fc3(out)

        return out
