import pandas as pd
from deepchem.feat.smiles_tokenizer import SmilesTokenizer

data = pd.read_csv(r'data\input_smiles.csv')

vocab = r'data\vocab.txt'

smiles = list(data['SMILES'])

tokenizer = SmilesTokenizer(vocab)

tokens = list()
for smile in smiles:
    tokens.append(tokenizer.encode(smile))

data['tokenized'] = tokens
data.to_csv(r'data\input_smiles_tokens.csv')
