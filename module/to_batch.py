#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch


def pad_collate(batch):

    seqs_labels = np.array(batch)[:, 1]

    vectorized_seqs = np.array(batch)[:, 0]
    seq_lengths = torch.LongTensor([len(x) for x in vectorized_seqs])

    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    vectorized_seqs = np.array(seq_tensor)

    return torch.tensor(vectorized_seqs), torch.LongTensor([x for x in
                                                            seqs_labels])
