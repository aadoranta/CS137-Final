import sys
import os
import pandas as pd
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer

sys.path.insert(0, r"{}\vq-vae\sonnet\sonnet\src\nets".format(os.getcwd()))

from vqvae import VectorQuantizer

data = pd.read_csv(r'data\input_smiles.csv')

vocab = r'data\vocab.txt'

def basic_encoder(input_data):

    """
    :param input_data: Dataframe containing smiles strings
    :return: Disambiguation of smiles strings into discrete elements (in string format)
    """

    tokenizer = BasicSmilesTokenizer()
    smiles = list(input_data['SMILES'])

    tokens = list()
    for smile in smiles:
        tokens.append(tokenizer.tokenize(smile))

    return tokens


def simple_encoder(input_data, input_vocab):

    """
    :param input_data: Dataframe containing smiles strings
    :param input_vocab: Vocabulary for typical components of smiles strings
    :return: One-to-one mapping of smiles strings to int representations
    """

    smiles = list(input_data['SMILES'])

    tokenizer = SmilesTokenizer(input_vocab)

    tokens = list()
    for smile in smiles:
        tokens.append(tokenizer.encode(smile))

    return tokens

def vqvae_encoder():

    vecquant = VectorQuantizer()


if __name__ == "__main__":
    print(basic_encoder(data))
