import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tokenized_smiles import TokenizedSmiles, FullData
from vectorquantizer import VectorQuantizer
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
from encoder import Encoder
from decoder import Decoder

# Possibly use extra data from the other generated smiles strings to make this more robust

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data = pd.read_csv(r'data\updated_smiles.csv')
# vocab = r'data\vocab.txt'
#
# tokenizer = Tokenizer(data=data, vocab=vocab)
# tokenizer.int_encoder(pad=True, save_csv=True, path=r'data\updated_smiles_tokenized.csv')

tokenized_smiles = FullData(path=r'data\full_data_tokenized.csv', label='Metal_q')

# Should be divided by size of data
data_variance = 201.6591565127223 / 512 # np.var(tokenized_smiles.data)


class Model(nn.Module):
    def __init__(self, num_hiddens1, num_hiddens2, num_hiddens3, num_hiddens4, num_embeddings, embedding_dim, commitment_cost):
        super(Model, self).__init__()

        self._encoder = Encoder(512,
                                num_hiddens1,
                                num_hiddens2
                                )
        # self._pre_vq_conv = nn.Linear(num_hiddens2, embedding_dim)

        self._pre_vq_conv = nn.Conv1d(in_channels=1,
                                      out_channels=embedding_dim,
                                      kernel_size=3,
                                      stride=1, padding='same')

        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim*num_hiddens2,
                                num_hiddens3,
                                num_hiddens4,
                                )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity, quantized


num_hiddens1 = 256
num_hiddens2 = 64
num_hiddens3 = 64
num_hiddens4 = 512
embedding_dim = 32
num_embeddings = 256
commitment_cost = 0.25
learning_rate = 1e-3
num_training_updates = 15000
batch_size = 1
num_epochs = 15

smiles_loader = DataLoader(tokenized_smiles, batch_size=batch_size, shuffle=True)


model = Model(num_hiddens1=num_hiddens1,
              num_hiddens2=num_hiddens2,
              num_hiddens3=num_hiddens3,
              num_hiddens4=num_hiddens4,
              embedding_dim=embedding_dim,
              num_embeddings=num_embeddings,
              commitment_cost=commitment_cost,
              ).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

model.train()

for epoch in range(num_epochs):

    train_res_recon_error = []
    train_res_perplexity = []

    tokens = list()
    labels = list()
    quantizeds = list()

    for i, data in enumerate(smiles_loader):

        data, y = data
        data = torch.tensor(data, dtype=torch.float32)
        data = data.to(device)
        optimizer.zero_grad()
        model.zero_grad()

        vq_loss, data_recon, perplexity, quantized = model(data)
        # data_recon = torch.squeeze(data_recon)
        # data = torch.squeeze(data)
        # Should we weight the reconstruction loss higher?
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())
        data_sav = torch.squeeze(data)
        label_sav = torch.squeeze(y)
        q_sav = torch.flatten(quantized)
        tokens.append(list(data_sav.cpu().detach().numpy()))
        labels.append(list(y.cpu().detach().numpy()))
        quantizeds.append(list(q_sav.cpu().detach().numpy()))

        if (i + 1) % 1000 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-1000:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-1000:]))
            print()

    print("--------DONE EPOCH {}--------".format(epoch + 1))
    print(np.mean(train_res_recon_error))
    print(list(data))
    print(list(data_recon))
    torch.save(model, 'autoencoder_model-{}.pt'.format(epoch + 1))
    df = pd.DataFrame()
    df['tokens'] = tokens
    df['embeddings'] = quantizeds
    df['label'] = labels
    df.to_csv(r'data\embeddings\encoded_{}.csv'.format(epoch))
