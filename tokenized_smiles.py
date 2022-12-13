import pandas as pd
import numpy as np
import torch
import ast
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

class FullData(Dataset):

    def __init__(self, path, label, transform=None, target_transform=None):
        print("STARTED LOADING FROM {}".format(path))
        self.index_dict = {'Electronic_E': 8, 'Dispersion_E': 9, 'Dipole_M': 10,
                           'Metal_q': 11, 'HL_Gap': 12, 'HOMO_Energy': 13,
                           'LUMO_Energy': 14, 'Polarizability': 15}
        self.df = pd.read_csv(path)
        self.x = torch.tensor(np.array([np.array(ast.literal_eval(row[-1])) for row in self.df.to_numpy()]))
        self.y = torch.tensor(np.array([np.array(row[self.index_dict[label]]) for row in self.df.to_numpy()]))
        self.transform = transform
        self.target_transform = target_transform
        print("DONE LOADING FROM {}".format(path))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        X = self.x[idx]
        Y = self.y[idx]

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            X = self.target_transform(X)

        return X, Y
