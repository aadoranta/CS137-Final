import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tokenized_smiles import TokenizedSmiles
from vectorquantizer import VectorQuantizer
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
from encoder import Encoder
from decoder import Decoder

# Possibly use extra data from the other generated smiles strings to make this more robust

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv(r'data\updated_smiles.csv')
vocab = r'data\vocab.txt'

tokenizer = Tokenizer(data=data, vocab=vocab)
tokenizer.int_encoder(pad=True, save_csv=True, path=r'data\updated_smiles_tokenized.csv')

tokenized_smiles = TokenizedSmiles(path=r'data\updated_smiles_tokenized.csv')

# Should be divided by size of data
data_variance = 1 # np.var(tokenized_smiles.data)


class Model(nn.Module):
    def __init__(self, num_hiddens, num_embeddings, embedding_dim, commitment_cost):
        super(Model, self).__init__()

        self._encoder = Encoder(1,
                                num_hiddens,
                                )
        self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1, padding='same')

        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                )

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity


num_hiddens = 32
embedding_dim = 64
num_embeddings = 128
commitment_cost = 0.25
learning_rate = 1e-3
num_training_updates = 15000
batch_size = 4
num_epochs = 10

smiles_loader = DataLoader(tokenized_smiles, batch_size=batch_size, shuffle=True)


model = Model(num_hiddens=num_hiddens,
              embedding_dim=embedding_dim,
              num_embeddings=num_embeddings,
              commitment_cost=commitment_cost,
              ).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

model.train()
train_res_recon_error = []
train_res_perplexity = []

for _ in range(num_epochs):
    for i, data in enumerate(smiles_loader):

        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        # Should we weight the reconstruction loss higher?
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i + 1) % 1000 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-1000:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-1000:]))
            print()

    print(list(data))
    print(list(data_recon))
