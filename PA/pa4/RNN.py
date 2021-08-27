from unicodedata import bidirectional
from nltk.sem.evaluate import _TUPLES_RE
import torch
import torch.nn as nn


class RNN_M2O(torch.nn.Module):

    def __init__(self, args, vocab_size):
        super(RNN_M2O, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = 150
        self.hidden_dim = args.hidden_dim
        self.layer_num = args.layer_num
        self.dropout = args.dropout
        self.batch_size = args.batch_size

        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.rnn = nn.RNN(
            input_size = self.embed_dim, 
            hidden_size = self.hidden_dim, 
            num_layers = self.layer_num,
            bidirectional = False,
            batch_first = True)

        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(self.hidden_dim)
        self.readout = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, h):
        h = self.embedding(h)  # (b,5) -> (b,5,emb)

        h = h.transpose(0, 1)  # (b,5,emb) -> (5,b,emb)
        h, _ = self.rnn(h, None)  # (5,b,emb) -> (5,b,hidden)
        h = h[4]  # (b,hidden) pick the last output

        h = self.dropout(h)
        h = self.layernorm(h)
        h = self.readout(h)  # (b,hidden) -> (b,vocab_size)

        return h
