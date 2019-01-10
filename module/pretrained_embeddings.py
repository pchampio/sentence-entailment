#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import torch
from torchtext import vocab


def load_embedding(sick_dataset_train, embeddings_size=50,
                   vocabulary_size=1000):
    # vocab is shared across all the text fields
    # CAUTION: GloVe will download all embeddings locally (862 MB).
    pretrained_emb = vocab.GloVe(name='6B', dim=embeddings_size)

    # 0 is for the padding when using mini-batch (start at one, shift by one)
    # do not forget the unk
    weights_matrix = np.zeros((vocabulary_size + 2, embeddings_size))

    found = 0
    no_found = 0
    # build a matrix of weights that will be loaded into
    # the PyTorch embedding layer
    for word_id in sick_dataset_train.dictionary:
        word = sick_dataset_train.dictionary[word_id]
        if word in pretrained_emb.stoi:
            pretrained_emb_ID = pretrained_emb.stoi[word]

            weights_matrix[word_id+1] = \
                pretrained_emb.vectors[pretrained_emb_ID]
            found += 1
        else:
            weights_matrix[word_id+1] = np.random.normal(scale=0.6,
                                                         size=(embeddings_size,
                                                               ))
            no_found += 1

    # UNK
    id_unk = pretrained_emb.stoi['unk']
    weights_matrix[vocabulary_size+1] = pretrained_emb.vectors[id_unk]

    pretrained_emb_vec = torch.tensor(weights_matrix, dtype=torch.float32)

    print("Voc that was found in the pretrained embs: " + str(found))
    print("Voc that wasn't found in the pretrained embs: " + str(no_found))
    print("Downloaded: Pretained Embedding matrix: " +
          str(pretrained_emb.vectors.size()))
    print("Adapted:    Pretained Embedding matrix: " +
          str(pretrained_emb_vec.size()))
    print()
    return pretrained_emb_vec
