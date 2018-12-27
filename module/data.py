#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Sentences loading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from gensim import corpora

class SickDataset(Dataset):
    endOfSentence   = '</s>'
    startOfSentence = '<s>'
    separator2Sentences = '<sep>'

    text_label = ["NEUTRAL", "ENTAILMENT", "CONTRADICTION"]

    tokens = [startOfSentence, separator2Sentences, endOfSentence]

    def join_sentence(self, row):
        """
        Create a new sentence (<s> + s_A + <sep> + s_B + </s>)
        """
        sentence_a = row['sentence_A'].split(" ")
        sentence_b = row['sentence_B'].split(" ")
        return np.concatenate((
            [self.startOfSentence],
            sentence_a,
            [self.separator2Sentences],
            sentence_b,
            [self.endOfSentence]
        ))

    def series_text_2_labelID(self, series, keep_n=1000):
        """
        Convert text Label into label id
        """
        reverse_dict = {v: k for k, v in  dict(enumerate(self.text_label)).items()}
        return series.map(reverse_dict)

    def series_2_dict(self, series, keep_n):
        """
        Convert document (a list of words) into a list of indexes
        AND apply some filter on the documents
        """
        dictionary = corpora.Dictionary(series)
        dictionary.filter_extremes(
            no_below=1,
            no_above=1,
            keep_n=keep_n,
            keep_tokens=self.tokens)
        return dictionary


    def __init__(self, df, vocabulary_size, dic=None):
        self.vocabulary_size = vocabulary_size

        # Label text as ids
        df["entailment_id"] = self.series_text_2_labelID(df['entailment_judgment'])

        # Add <s>,</s>,<sep> tokens to the vocabulary
        df['sentence_AB'] = df.apply(self.join_sentence, axis=1)

        # check if the dictionary is given
        if dic is None:
            # Create the Dictionary
            self.dictionary = self.series_2_dict(df['sentence_AB'], vocabulary_size)
        else:
            self.dictionary = dic

        # sentence of words -> array of idx
        # Adds unknown to the voc (idx = len(dictionary)), len(dictionary) = vocabulary_size
        # Adds one to each (no tokens at 0, even <unk>)
        # 0 is for the padding when using mini-batch
        df["word_idx"] = df["sentence_AB"].apply(
            lambda word: np.array(self.dictionary.doc2idx(word, unknown_word_index=vocabulary_size)) + 1
        )

        self.df = df

        # compute a sorted occurence dictionary on the whole corpus
        occ_dict = {}
        for serie in df['sentence_AB']:
            unique, counts = np.unique(serie, return_counts=True)
            tmp_dict = dict(zip(unique, counts))

            for key, value in tmp_dict.items():
                if key in occ_dict:
                    occ_dict[key] = occ_dict[key] + tmp_dict[key]
                else:
                    occ_dict[key] = value

        self.occ_dict_list = [[key, value] for key, value in occ_dict.items()]
        self.occ_dict_list.sort(key=lambda x: x[1], reverse=True)

    def getSortedOccDictList(self):
        return self.occ_dict_list

    def plotVocabularyCoverage(self):
        occdict_list = self.occ_dict_list

        total = 0
        y = []
        for i, value in enumerate(occdict_list):
            total += value[1]
            y.append(total)
            if (i == self.vocabulary_size):
                current_voc_cov = total

        current_voc_cov = current_voc_cov*100.0/total

        y = [tmp*100.0/total for tmp in y]

        x = np.linspace(0, len(occdict_list), len(occdict_list))

        # Show graph
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 20
        fig_size[1] = 9
        plt.rcParams["figure.figsize"] = fig_size

        legend, = plt.plot(x, y, label='Vocabulary size ')

        plt.title(('Current vocabulary size n=' + str(self.vocabulary_size) + ' coverage = ' +"{:.4}".format(current_voc_cov) + '%'),
                     fontsize=14, fontweight='bold', color='gray')
        plt.suptitle(('Vocabulary coverage'),
                     fontsize=24, fontweight='bold', color='gray')
        plt.xlabel("Size of unique vocabulary", color='gray', fontsize=14)
        plt.ylabel("Vocabulary coverage %", color='gray', fontsize=14)

        ## Plot Swagg ##
        plt.yticks(fontsize=14, rotation=0, color='gray')
        plt.xticks(fontsize=14, rotation=0, color='gray')

        # Less border
        plt.gca().xaxis.grid(True)
        plt.gca().yaxis.grid(True)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.show()

    def getRef(self, index):
        return self.df['sentence_AB'][index]

    def __getitem__(self, index):
        return (
            self.df['word_idx'][index],
            self.df['entailment_id'][index])

    def getDictionary(self):
        return self.dictionary

    def __len__(self):
        return len(self.df)

