#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
import torch
from einops import rearrange, reduce


def _vprint(*args, **kwargs): None


class RNNClassifierBase(nn.Module):
    def __init__(self, input_voc_size, embedding_size,
                 device="cpu"):
        super(RNNClassifierBase, self).__init__()

        self.input_voc_size = input_voc_size
        self.embedding_size = embedding_size
        self.hidden_size = 20
        self.rnn_out_size = self.hidden_size * 2
        self.device = device

        self.num_classes = 3

        # Add the padding token (0) (+1 to voc_size)
        # Pads the output with the embedding vector at padding_idx whenever it
        # encounters the index..
        self.embedding = nn.Embedding(input_voc_size+1, embedding_size,
                                      padding_idx=0)
        # Load the pretrained embeddings
        # self.embedding.weight = nn.Parameter(pretrained_emb_vec)
        # embeddings fine-tuning
        self.embedding.weight.requires_grad = False

        self.rnn = nn.GRU(
              input_size=embedding_size,
              hidden_size=self.hidden_size,
              batch_first=True,
              bidirectional=True,
        )

        self.fc1 = nn.Linear(self.rnn_out_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    # input shape: B x S (input size)
    def forward(self, x, vprint=_vprint):

        vprint("\nsize input", x.size())

        batch_size = x.size(0)

        # Initialize hidden (num_layers * num_directions, batch_size,
        # hidden_size)
        h_0 = torch.zeros(2, batch_size, self.hidden_size)
        vprint("size hidden init", h_0.size())

        # When creating new variables inside a model (like the hidden state in
        # an RNN/GRU/LSTM),
        # make sure to also move them to the device (GPU or CPU).
        h_0 = h_0.to(self.device)

        # Embedding B x S -> B x S x I (embedding size)
        emb = self.embedding(x)
        vprint("size Embedding", emb.size())

        # Propagate embedding through RNN
        # Input: (batch, seq_len, embedding_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        out, hidden = self.rnn(emb, h_0)

        vprint("size hidden", hidden.size())

        rnn_out = torch.cat((hidden[0], hidden[1]), 1)
        vprint("size rnn out", rnn_out.size())

        # Use the last layer output as FC's input
        layout_fc1 = self.fc1(rnn_out)
        vprint("size layout fc1", layout_fc1.size())

        fc_output = self.softmax(layout_fc1)

        return fc_output


class RNNClassifierDouble(nn.Module):
    def __init__(self, input_voc_size, embedding_size,
                 device="cpu"):
        super(RNNClassifierDouble, self).__init__()

        self.input_voc_size = input_voc_size
        self.embedding_size = embedding_size
        self.hidden_size = 742
        self.device = device

        self.num_classes = 3

        # Add the padding token (0) (+1 to voc_size)
        # Pads the output with the embedding vector at padding_idx whenever it
        # encounters the index..
        self.embedding = nn.Embedding(input_voc_size+1, embedding_size,
                                      padding_idx=0)
        # Load the pretrained embeddings
        # self.embedding.weight = nn.Parameter(pretrained_emb_vec)
        # embeddings fine-tuning
        self.embedding.weight.requires_grad = False

        self.rnn = nn.GRU(
              input_size=embedding_size,
              hidden_size=self.hidden_size,
              batch_first=True,
              dropout=0.5,
              bidirectional=True,
        )

        self.conv = nn.Conv1d(
            in_channels=2,  # BiRNN
            out_channels=1012,
            kernel_size=self.hidden_size,
            stride=self.hidden_size
        )

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(1012, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    # input shape: B x S (input size)
    def forward(self, x, vprint=_vprint):

        # 2 Sentences
        x_a = x[0]
        x_b = x[1]

        batch_size = x_a.size(0)

        # Initialize hidden (num_layers * num_directions, batch_size,
        # hidden_size)
        h_0_a = torch.zeros(2, batch_size, self.hidden_size)
        h_0_b = torch.zeros(2, batch_size, self.hidden_size)
        vprint("size hidden init", h_0_a.size())

        # When creating new variables inside a model (like the hidden state in
        # an RNN/GRU/LSTM),
        # make sure to also move them to the device (GPU or CPU).
        h_0_a = h_0_a.to(self.device)
        h_0_b = h_0_b.to(self.device)

        # Embedding B x S -> B x S x I (embedding size)
        emb_a = self.embedding(x_a)
        emb_b = self.embedding(x_b)
        vprint("size Embedding", emb_a.size())

        # Propagate embedding through RNN
        # Input: (batch, seq_len, embedding_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        output_a, hidden_a = self.rnn(emb_a, h_0_a)
        output_b, hidden_b = self.rnn(emb_b, h_0_b)

        rearrange_ops = "batch seqlen (dir out) -> batch dir (seqlen out)"
        rearrange_a = rearrange(output_a, rearrange_ops, out=self.hidden_size)  # noqa: E501
        rearrange_b = rearrange(output_b, rearrange_ops, out=self.hidden_size)  # noqa: E501

        rea_ab = rearrange([rearrange_a, rearrange_b], "a b c d -> b c (a d)")

        conv = self.conv(rea_ab)
        vprint(conv.shape)

        pool = reduce(conv, "batch channels out -> batch channels", 'max')

        relu = self.relu(pool)
        layout_fc1 = self.fc1(relu)

        # Use the last layer output as FC's input
        vprint("size layout fc1", layout_fc1.size())

        fc_output = self.softmax(layout_fc1)

        return fc_output
