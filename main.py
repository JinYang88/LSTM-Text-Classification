
# coding: utf-8

# In[38]:

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
LOGGER = logging.getLogger("toxic_dataset")


# In[39]:

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"<br />",r" ",s)
    s = re.sub(r'(\W)(?=\1)', '', s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    
    return s


# In[40]:

device = 0 # 0 for gpu, -1 for cpu
batch_size = 16
test_mode = 0  # 0 for train+test 1 for test


# In[67]:

print('Reading data..')
normalize_pipeline = data.Pipeline(convert_token=normalizeString)
ID = data.Field(sequential=False, batch_first=True)
TEXT = data.Field(sequential=True, lower=True, eos_token='<EOS>', init_token='<BOS>',
                  pad_token='<PAD>', fix_length=None, batch_first=True, preprocessing=normalize_pipeline)
LABEL = data.Field(sequential=False, batch_first=True)


# In[68]:

train = data.TabularDataset(
        path='./data/train.tsv', format='tsv',
        fields=[('Id', ID), ('Label', LABEL), ('Review', TEXT)], skip_header=True)


# In[69]:

test = data.TabularDataset(
        path='./data/test.tsv', format='tsv',
        fields=[('Id', ID), ('Review', TEXT)], skip_header=True)


# In[72]:

train.examples[0].Review


# In[73]:

TEXT.build_vocab(train.Review,test.Review)
ID.build_vocab(train.Id, test.Id)
LABEL.build_vocab(train.Label, test.Label)


# In[87]:

train_iter = data.BucketIterator(dataset=train, batch_size=batch_size, sort_key=lambda x: len(x.Review), device=device, repeat=False)


# In[88]:

train_iter.data()[0].Review


# In[90]:

test_iter = data.Iterator(dataset=test, batch_size=batch_size, device=device, shuffle=False, repeat=False)
print('Reading data done.')


# In[91]:

class Model(torch.nn.Module) :
    def __init__(self,vocab_size, embedding_dim, hidden_dim, batch_size) :
        super(Model,self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linearOut = nn.Linear(hidden_dim,2)
    def forward(self,inputs) :
        x = self.embeddings(inputs)
        lstm_out,lstm_h = self.lstm(x, None)
        x = lstm_out[:, -1, :]
        x = self.linearOut(x)
        x = F.log_softmax(x, dim=1)

        return x,lstm_h
    def init_hidden(self) :
        return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))  


# In[92]:

embedding_dim = 100
hidden_dim = 64
epochs = 4

print('Initialing model..')
model = Model(len(TEXT.vocab), embedding_dim, hidden_dim, batch_size)
if device == 0:
    model.cuda()


# In[93]:

if not test_mode:
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print('Start training..')

    train_iter.create_batches()
    batch_num = len(list(train_iter.batches))

    for i in range(epochs) :
        avg_loss = 0.0
        train_iter.init_epoch()
        batch_count = 0
        for batch in iter(train_iter):
            batch_start = time.time()
            y_pred,_ = model(batch.Review)
            loss = loss_function(y_pred, batch.Label-1)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1
            batch_end = time.time()
            print('Finish {}/{} batch, {}/{} epoch. Time consuming {}s, loss is {}'.format(batch_count, batch_num, i+1, epochs, round(batch_end - batch_start, 2), float(loss)))
        torch.save(model.state_dict(), 'model' + str(i+1)+'.pth')           

# Test
print('Start predicting...')
model.load_state_dict(torch.load('model{}.pth'.format(epochs)))

f1 = open('submission.csv','w')
f1.write('"id","sentiment"'+'\n')
final_res = []

for batch in iter(test_iter):
    hidden = model.init_hidden()
    y_pred,_ = model(batch.Review)
    pred_res = y_pred.data.max(1)[1].cpu().numpy()
    final_res.extend(pred_res)

print('Prediction done...')
for idx, res in enumerate(final_res):
    text_id = test_iter.dataset.examples[idx].Id
    f1.write(text_id + ',' + str(res)+'\n')
print('Results dumping done...')


# In[ ]:




# In[ ]:



