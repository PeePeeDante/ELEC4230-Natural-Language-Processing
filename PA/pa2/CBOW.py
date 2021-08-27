import torch
import torch.nn as nn
import torch.nn.functional as F

from loader import WIN_SIZE

class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.window_size = WIN_SIZE
    
    def forward(self,inputs):
        
        emb = self.embeddings(inputs)

        emb_sum = torch.sum(emb,dim=1)/4
        emb_sum = torch.squeeze(emb_sum)

        out2 = self.linear(emb_sum) # B * vocab_size
        #print(out2.shape)
        log_prob = F.log_softmax(out2, dim=1)
        
        return log_prob
        
    def get_word_embedding(self, wrd_vec):
        return self.embeddings(wrd_vec)
