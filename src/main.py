
# coding: utf-8
import pandas as pd
import numpy as np
import re
import logging
import torch
from torchtext import data
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
import io
import time
import sys
import model
import datahelper


device = -1 # 0 for gpu, -1 for cpu
batch_size = 16
test_mode = 0  # 0 for train+test 1 for test
embedding_dim = 100
hidden_dim = 64
epochs = 4
start_with = 4 # start at 4th epoch



print('Reading data..')
normalize_pipeline = data.Pipeline(convert_token=datahelper.normalizeString)
ID = data.Field(sequential=False, batch_first=True)
TEXT = data.Field(sequential=True, lower=True, eos_token='<EOS>', init_token='<BOS>',
                  pad_token='<PAD>', fix_length=None, batch_first=True, preprocessing=normalize_pipeline)
LABEL = data.Field(sequential=False, batch_first=True)


train = data.TabularDataset(
        path='../data/train.tsv', format='tsv',
        fields=[('Id', ID), ('Label', LABEL), ('Review', TEXT)], skip_header=True)
test = data.TabularDataset(
        path='../data/test.tsv', format='tsv',
        fields=[('Id', ID), ('Review', TEXT)], skip_header=True)


TEXT.build_vocab(train.Review,test.Review)
ID.build_vocab(train.Id, test.Id)
LABEL.build_vocab(train.Label, test.Label)


print('Build Finished.')

train_iter = data.BucketIterator(dataset=train, batch_size=batch_size, sort_key=lambda x: len(x.Review), device=device, repeat=False)
test_iter = data.Iterator(dataset=test, batch_size=batch_size, device=device, shuffle=False, repeat=False)


train_dl = datahelper.BatchWrapper(train_iter, "Review", ["Label"])
test_dl = datahelper.BatchWrapper(test_iter, "Review", ["Id"])

print('Reading data done.')



print('Initialing model..')
MODEL = model.bi_lstm(len(TEXT.vocab), embedding_dim, hidden_dim, batch_size)
if start_with > 0:
    MODEL.load_state_dict(torch.load('model{}.pth'.format(start_with)))

if device == 0:
    MODEL.cuda()

# Train
if not test_mode:
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(MODEL.parameters(), lr=1e-3)
    print('Start training..')

    train_iter.create_batches()
    batch_num = len(list(train_iter.batches))
    batch_start = time.time()

    for i in range(epochs) :
        avg_loss = 0.0
        train_iter.init_epoch()
        batch_count = 0
        for batch, label in train_dl:
            y_pred = MODEL(batch)
            loss = loss_function(y_pred, label-1)
            MODEL.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1
            if batch_count % 50 == 0:
                batch_end = time.time()
                print('Finish {}/{} batch, {}/{} epoch. Time consuming {}s, loss is {}'.format(batch_count, batch_num, i+1, epochs, round(batch_end - batch_start, 2), float(loss)))
        torch.save(MODEL.state_dict(), 'model' + str(i+1)+'.pth')           

# Test
print('Start predicting...')
MODEL.load_state_dict(torch.load('model{}.pth'.format(epochs)))

f1 = open('submission.csv','w')
f1.write('"id","sentiment"'+'\n')
final_res = []

for batch, _ in iter(test_dl):
    y_pred = MODEL(batch)
    pred_res = y_pred.data.max(1)[1].cpu().numpy()
    final_res.extend(pred_res)

print('Prediction done...')
for idx, res in enumerate(final_res):
    text_id = test_iter.dataset.examples[idx].Id
    f1.write(text_id + ',' + str(res)+'\n')
print('Results dumping done...')









