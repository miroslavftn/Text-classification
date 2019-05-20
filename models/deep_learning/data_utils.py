import csv
import os.path as osp
import pickle

import numpy as np
import pandas as pd
from configs import DATA_DIR


class DataUtils:
    """
    Helper class to load and prepared data for training.
    """
    @staticmethod
    def save_tokenizer(tokenizer, path):
        try:
            with open(path, "wb") as f:
                pickle.dump(tokenizer, f)
        except:
            print("Saving tokenizer on path {} was unsuccessful!".format(path))


    @staticmethod
    def load_tokenizer(path):
        try:
            with open(path, "rb") as f:
                tokenizer = pickle.load(f)
                tokenizer.oov_token = None
                return tokenizer
        except:
            print("Couldn't load tokenizer from path `{}`!".format(path))
            return None

    @staticmethod
    def save_embedding_matrix(embedding_matrix, path):
        try:
            np.save(path, embedding_matrix)
        except:
            print("Saving embedding matrix was unsuccessful!")

    @staticmethod
    def load_embedding_matrix(emb_matrix_path=None, tokenizer=None):
        """
        Firstly tries to load embedding matrix if the file is available. If it's not, build new embedding matrix from
        vocabulary and save it for further use.

        Parameters
        ----------
        emb_matrix_path : str
        tokenizer : keras.Tokenizer

        Returns
        -------
        embedding_matrix : np.array
            Numpy 2D-array (matrix) containing pre-trained vectors for words in vocabulary.
        """
        embedding_matrix = None

        if emb_matrix_path:
            try:
                embedding_matrix = np.load(emb_matrix_path)
            except:
                print("Couldn't load embedding matrix on local path: {}.".format(emb_matrix_path))
        if tokenizer:
            try:
                embedding_matrix = Glove().build_embedding_matrix(word2idx=tokenizer.word_index)
            except:
                print("Couldn't build embedding matrix with tokenizer")

        return embedding_matrix


class Glove:
    def __init__(self, glove_file='glove.6B.300d.txt'):
        self.glove_path = osp.join(DATA_DIR, glove_file)
        self.glove = self.read_glove()
        self.embedding_dim = self.get_embedding_dim()

    def read_glove(self):
        """
        Reads pre-trained glove vectors from `glove.6B.300d.txt`. This file is required!

        Returns
        -------
        pandas.DataFrame
            Glove pre-trained weights
        """
        try:
            glove = pd.read_table(self.glove_path, sep=" ", header=None, quoting=csv.QUOTE_NONE)
        except FileNotFoundError:
            print("Glove file not found, please make sure you have it downloaded and placed in "
                  "`DATA_DIR` directory!")
            glove = None
        return glove

    def get_embedding_dim(self):
        """
        Calculate embedding dimension (equals to number of columns)

        Returns
        -------
        embedding_dim : int
        """
        embedding_dim = None if self.glove is None else self.glove.shape[1] - 1

        return embedding_dim

    def build_embedding_matrix(self, word2idx):
        """
        Create embedding weights for Keras Embedding layer.

        Parameters
        ----------
        word2idx : dict
            Words in vocabulary for which to look up weight vectors.

        Returns
        -------
        numpy array
            Embedding matrix where each row/index correspond to word in `word2idx` and its vector weights.
        """
        if self.glove is not None:
            words = self.glove.rename(columns={self.glove.columns[0]: "words"})
            words['words'].apply(str)

            # Convert to dictionary
            embedding_index = words.set_index('words').T.to_dict('list')
            # i=0 -> PAD; i=len(word2idx)+1 -> UNK;
            embedding_matrix = np.zeros((len(word2idx) + 2, self.embedding_dim))

            # Create embedding matrix
            for word, i in word2idx.items():
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        else:
            print("Trying to build embedding matrix build glove is not loaded!")
            embedding_matrix = None

        return embedding_matrix

