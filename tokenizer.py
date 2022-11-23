import pandas as pd
from deepchem.feat.smiles_tokenizer import SmilesTokenizer

data = pd.read_csv(r'data\input_smiles.csv')

vocab = r'data\vocab.txt'

smiles = list(data['SMILES'])

tokenizer = SmilesTokenizer(vocab)
print(tokenizer.get_vocab())

for smile in smiles:
    print(tokenizer.encode(smile))
