import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer
from sonnet.nets import VectorQuantizer


class Tokenizer:

    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def string_encoder(self):

        """
        :param input_data: Dataframe containing smiles strings
        :return: Disambiguation of smiles strings into discrete elements (in string format)
        """

        tokenizer = BasicSmilesTokenizer()
        smiles = list(self.data['SMILES'])

        tokens = list()
        for smile in smiles:
            tokens.append(tokenizer.tokenize(smile))

        return tokens


    def int_encoder(self, pad=False):

        """
        :param input_data: Dataframe containing smiles strings
        :param input_vocab: Vocabulary for typical components of smiles strings
        :return: One-to-one mapping of smiles strings to int representations
        """

        smiles = list(self.data['SMILES'])

        tokenizer = SmilesTokenizer(self.vocab)

        tokens = list()
        for smile in smiles:
            tokens.append(torch.tensor(tokenizer.encode(smile), dtype=torch.float32))

        if pad:
            tokens = pad_sequence(tokens, batch_first=True, padding_value=-1)

        return tokens


def vqvae_encoder(input_data):

    smiles = list(input_data['SMILES'])

    vec_quantize = VectorQuantizer(embedding_dim=16, num_embeddings=1024, commitment_cost=0.1)

    for smile in smiles:
        print(vec_quantize.quantize(smile))

