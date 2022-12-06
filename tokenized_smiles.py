# Use 1D convolution to decrease dimensionality into embedding size

import pandas as pd
from tokenizer import Tokenizer
from torch.utils.data import Dataset

data = pd.read_csv(r'data\updated_smiles.csv')
vocab = r'data\vocab.txt'

tokenizer = Tokenizer(data=data, vocab=vocab)
tokenizer.int_encoder(pad=True, save_csv=True, path=r'data\updated_smiles_tokenized.csv')


class TokenizedSmiles(Dataset):

    def __init__(self, path, transform=None, target_transform=None):
        self.data = list(pd.read_csv(path))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]

        if self.transform:
            smiles = self.transform(smiles)
        if self.target_transform:
            smiles = self.target_transform(smiles)

        return smiles

