#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from module.utils import force_symlink
from module.evaluate import evaluate
import torch
import time
from tensorboardX import SummaryWriter

# #----------------------------------------#
#  Change import to change the architecture
# #----------------------------------------#

#  from module.data import SickDatasetBase as SickDataset
#  from module.to_batch import pad_collate_single_sentence as pad_collate
#  from module.models import RNNClassifierBase as RNNClassifier

from module.data import SickDatasetDouble as SickDataset
from module.to_batch import pad_collate_double_sentence as pad_collate
from module.models import RNNClassifierDouble as RNNClassifier

# #----------------------------------------#

from module.pretrained_embeddings import load_embedding

from torch.utils.data import DataLoader

pd.set_option("display.width", 280)
pd.set_option('max_colwidth', 50)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS = 100
BATCH_SIZE = 8
VOCABULARY_SIZE = 1500


# ## Board ###
writer = SummaryWriter()

#######################
#  Load the datasets  #
#######################


df_train = pd.read_csv("./sick_train/SICK_train.txt", sep="\t")
df_train = df_train.drop(['relatedness_score'], axis=1)

df_dev = pd.read_csv("./sick_trial/SICK_trial.txt", sep="\t")
df_dev = df_dev.drop(['relatedness_score'], axis=1)

df_test = pd.read_csv("./sick_test/SICK_test.txt", sep="\t")
df_test = df_test.drop(['relatedness_score'], axis=1)

# Create the train dataset
sick_dataset_train = SickDataset(df_train, VOCABULARY_SIZE)
#  print(sick_dataset_train.df.head())

dictionary_train = sick_dataset_train.getDictionary()

# Create the dev dataset
sick_dataset_dev = SickDataset(df_dev, VOCABULARY_SIZE, dictionary_train)
# Create the test dataset
sick_dataset_test = SickDataset(df_test, VOCABULARY_SIZE, dictionary_train)

sick_dataset_train.pprint()

#  sick_dataset_train.plotVocabularyCoverage()

#####################
#  Pretrained Embs  #
#####################
embeddings_size = 300


print()

pretrained_emb_vec = load_embedding(
    sick_dataset_train,
    embeddings_size=embeddings_size,
    vocabulary_size=VOCABULARY_SIZE)


# Debug
#  print(sick_dataset_train.dictionary.doc2idx(["the", "The"]))
#  print(sick_dataset_train.dictionary[18])
#  print(pretrained_emb_vec[18+1])
# Glove dim=50 word=the vector[:4] = 0.418 0.24968 -0.41242 0.1217

################
#  DataLoader  #
################

train_loader = DataLoader(dataset=sick_dataset_train,
                          batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=pad_collate
                          )

dev_loader = DataLoader(dataset=sick_dataset_dev,
                        batch_size=1, shuffle=False, collate_fn=pad_collate)

test_loader = DataLoader(dataset=sick_dataset_test,
                         batch_size=1, shuffle=False)

# Debug the padding
#  print([x for x in enumerate(train_loader)][0])
print()

################
#  Classifier  #
################

# Add the unknown token (+1 to voc_size)
rnn = RNNClassifier(VOCABULARY_SIZE+1, embeddings_size, 200, device=device)
rnn.to(device)
print(rnn)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
weights = [1-((sick_dataset_train.df['entailment_id'] == i).sum() /
              len(sick_dataset_train)) for i in range(3)]
class_weights = torch.FloatTensor(weights).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

##########
#  Loop  #
##########

iter = 0
iter_batch = 0
best_accuracy_dev = 0.0

rnn.train()
time_start = time.perf_counter()

for epoch in range(NUM_EPOCHS):
    total_correct = 0
    total_target = 0
    train_loss_batches = 0
    train_loss_batches_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        output = rnn(data)
        #  output = rnn(data, print)

        loss = criterion(output, target)

        train_loss_batches += loss.cpu().detach().numpy()
        train_loss_batches_count += 1

        rnn.zero_grad()
        loss.backward()
        optimizer.step()

        # Get the Accuracy
        _, predicted = torch.max(output.data, dim=1)
        correct = (predicted == target).sum().item()

        total_correct += correct
        total_target += target.size(0)

        if batch_idx % 200 == 0 or batch_idx % 200 == 1 or \
                batch_idx == len(train_loader)-1:
            print(('\rEpoch [{:3}/{}] | Step [{:5}/{} ({:3.0f}%)] |'
                   ' Loss {:.3f} | Accuracy {:.2f}%').format(
                    epoch+1, NUM_EPOCHS,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),
                    (total_correct / total_target) * 100), end=' ')

            writer.add_scalar('data/loss/_train_only', train_loss_batches /
                              train_loss_batches_count, iter_batch)
            iter_batch += 1

        if False:
            break

    accuracy_dev, loss_dev = evaluate(rnn, dev_loader, criterion=criterion,
                                      whileTraining=True, device=device)

    print("@ Loss_dev {:.3f} | Accuracy_dev {:.2f}%".format(loss_dev,
                                                            accuracy_dev))

    if best_accuracy_dev < accuracy_dev:
        best_accuracy_dev = accuracy_dev
        file = 'checkpoint.pth.' + str(epoch) + '.acc.' + \
            str(round(accuracy_dev, 2)) + '.tar'

        torch.save({
            'epoch': epoch+1,
            'model_state_dict': rnn.state_dict(),
            'loss': loss,
            }, file)
        force_symlink(file, 'checkpoint.pth.best.tar')
    writer.add_scalars('data/loss/evol', {'train': train_loss_batches /
                                          train_loss_batches_count,
                                          'dev': loss_dev}, iter)
    iter += 1

time_elapsed = (time.perf_counter() - time_start)
print("Learning finished!\n - in", round(time_elapsed, 2), "s")


##########
#  Eval  #
##########

checkpoint = torch.load('checkpoint.pth.best.tar')
rnn.load_state_dict(checkpoint['model_state_dict'])
print("=> loaded checkpoint epoch {}"
      .format(checkpoint['epoch']))


evaluate(rnn, dev_loader, writer=writer, device=device)


# evaluate(rnn, test_loader)
