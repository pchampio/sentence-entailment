#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch


def pad_vec(vectorized_seqs, pad_len=0):
    seq_lengths = torch.LongTensor([len(x) for x in vectorized_seqs])

    pad_len = max(pad_len, seq_lengths.max())

    seq_tensor = torch.zeros((len(vectorized_seqs), pad_len)).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    return np.array(seq_tensor)


def pad_collate_single_sentence(batch):
    seqs_labels = np.array(batch)[:, 1]

    vectorized_seqs = pad_vec(np.array(batch)[:, 0])

    return torch.tensor(vectorized_seqs), torch.LongTensor([x for x in
                                                            seqs_labels])


def find_double_sentence_pad_len(batch):
    vectorized_seqs_a = np.array(batch)[:, 0]
    vectorized_seqs_b = np.array(batch)[:, 1]

    seq_lengths_a = torch.LongTensor([len(x) for x in vectorized_seqs_a])
    seq_lengths_b = torch.LongTensor([len(x) for x in vectorized_seqs_b])

    return max(seq_lengths_b.max(), seq_lengths_a.max())


def pad_collate_double_sentence(batch):
    seqs_labels = np.array(batch)[:, 2]

    pad_len = find_double_sentence_pad_len(batch)

    vectorized_seqs_a = pad_vec(np.array(batch)[:, 0], pad_len=pad_len)
    vectorized_seqs_b = pad_vec(np.array(batch)[:, 1], pad_len=pad_len)

    return (torch.tensor([vectorized_seqs_a, vectorized_seqs_b]),
            torch.LongTensor([x for x in seqs_labels]))
