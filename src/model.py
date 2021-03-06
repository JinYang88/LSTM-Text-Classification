import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class lstm(torch.nn.Module) :
    def __init__(self,vocab_size, embedding_dim, hidden_dim, batch_size) :
        super(lstm,self).__init__()
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
        return x

    
class bi_lstm(torch.nn.Module) :
    def __init__(self,vocab_size, embedding_dim, hidden_dim, batch_size) :
        super(bi_lstm,self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, batch_first=True, bidirectional=True, dropout=0.8)
        self.linearOut = nn.Linear(hidden_dim,2)

    def forward(self,inputs) :
        x = self.embeddings(inputs)
        lstm_out,lstm_h = self.lstm(x, None)
        x = lstm_out[:, -1, :]
        x = self.linearOut(x)
        x = F.log_softmax(x, dim=1)
        return x
