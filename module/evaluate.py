#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_fscore_support

import itertools
import io
import matplotlib.pyplot as plt

import torch
import numpy as np
import pandas as pd

from module.data import SickDataset

def confusion_scores(total_labels, total_pred, writer=None):
    fig = plt.figure(figsize=(10,10))
    classes = SickDataset.text_label
    title='Confusion matrix'

    cm = confusion_matrix(total_labels, total_pred, labels=[0, 1, 2])

    plt.rcParams["figure.figsize"] = [10, 10]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, color='gray', fontsize=24)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, [c.lower() for c in classes], rotation=45 , style='italic', color='gray', fontsize=17)
    plt.yticks(tick_marks, [c.lower() for c in classes], color='gray', style='italic', fontsize=17)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', color='gray', fontsize=19)
    plt.xlabel('Predicted label', color='gray', fontsize=19)
    plt.tight_layout()
    plt.show()
    if writer != None:
        writer.add_figure('plt/confusion_matrix', fig, 0)


def evaluate(model, loader, whileTraining=False, criterion=None, writer=None, device="cpu"):
    """
    Displays the confusion_matrix the precision recall fscore
    If in whileTrainnig Mode only return the accuracy and loss
    """
    model.eval()
    with torch.no_grad():
        total_labels = torch.LongTensor([])
        total_pred = torch.LongTensor([])
        train_loss_batches = 0
        train_loss_batches_count = 0
        for batch_idx, (data, target) in enumerate(loader):

                data = data.to(device)
                target = target.to(device)

                output = model(data)

                if whileTraining and criterion != None:
                    loss = criterion(output, target)
                    train_loss_batches +=loss.cpu().detach().numpy()
                    train_loss_batches_count += 1

                # Get the Accuracy
                _, predicted = torch.max(output.data, dim=1)
                correct = (predicted == target).sum().item()

                total_labels = torch.cat((total_labels, target.cpu()))
                total_pred = torch.cat((total_pred, predicted.cpu()))


        model.train()
        if whileTraining and criterion!=None:
            return ((accuracy_score(total_labels.flatten().numpy(), total_pred.flatten().numpy()) * 100), train_loss_batches / train_loss_batches_count)


        confusion_scores(total_labels, total_pred, writer=writer)

        print("Accuracy:  {:.4f}".format(accuracy_score(total_labels, total_pred)))

        # compute per-label precisions, recalls, F1-scores, and supports instead of averaging
        metrics = precision_recall_fscore_support(
                                        total_labels, total_pred,
                                        average=None, labels=[0, 1, 2])

        df = pd.DataFrame(list(metrics), index=['Precision', 'Recall', 'Fscore', 'support'],
                                   columns=SickDataset.text_label)
        df = df.drop(['support'], axis=0)
        print(df.T)
