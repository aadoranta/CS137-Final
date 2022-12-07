import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class TokenizedSmiles(Dataset):

    def __init__(self, path, transform=None, target_transform=None):
        print("STARTED LOADING FROM {}".format(path))
        self.data = torch.Tensor(np.array([[i] for i in pd.read_csv(path).to_numpy()]))
        self.transform = transform
        self.target_transform = target_transform
        print("DONE LOADING FROM {}".format(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]

        if self.transform:
            smiles = self.transform(smiles)
        if self.target_transform:
            smiles = self.target_transform(smiles)

        return smiles

