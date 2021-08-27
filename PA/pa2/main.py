from CBOW import CBOW
from loader import MyData, BSZ, FILE_NAME
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

import numpy as np

EMBEDDING_DIM = 400
LR = 1e-1
NUM_EPOCHS = 20

def similarity(vec1,vec2):
    vec1 = torch.squeeze(vec1)
    vec2 = torch.squeeze(vec2)

    return (vec1 @ vec2) / (torch.norm(vec1) * torch.norm(vec2))

dataset = MyData()
print("========== Data info ==========")
print("Vocab size: ", dataset.vocab_count)
print("Sample size (target, context): ", dataset.len)
print("==========           ==========")

train_loader = DataLoader(dataset=dataset, batch_size=BSZ, shuffle=False)
model = CBOW(dataset.vocab_count, EMBEDDING_DIM)
loss_function = nn.NLLLoss()

if torch.cuda.is_available():
    model = model.cuda()
    loss_function = loss_function.cuda()
    
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

print("Start Training...")
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    start_time = time.time()
    batch_num = 1
    print("epoch ", epoch, " start")
        
    for i, (target, context) in enumerate(train_loader):

        t_class = target
        c_class = torch.stack(context)
        c_class = torch.transpose(c_class,0,1)
        
        if torch.cuda.is_available():
            c_class = c_class.cuda()
            t_class = t_class.cuda()
        
        if (t_class.shape[0] == 1):
            continue
            
        output = model(c_class)
        loss = loss_function(output, t_class)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print('batch:', batch_num, "\t| time {0:.2f}:", time.time() - start_time,"\t| batch loss:", loss.item())
        batch_num+=1
            
    print("epoch loss:",total_loss, "\t| time {0:.2f}:", time.time() - start_time)

print("Finish Training...")

word2vec = dataset.word2vec
print("Exporting word2vec.csv...")
import csv
with open('word2vec.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in word2vec.items():
        writer.writerow([key, value])

print("Similarity Test...")
emb = model.get_word_embedding

bad = torch.Tensor([word2vec["bad"]]).long().cuda()
nice = torch.Tensor([word2vec["nice"]]).long().cuda()
excellent = torch.Tensor([word2vec["excellent"]]).long().cuda()
poor = torch.Tensor([word2vec["poor"]]).long().cuda()
ugly = torch.Tensor([word2vec["ugly"]]).long().cuda()
beautiful = torch.Tensor([word2vec["beautiful"]]).long().cuda()
love = torch.Tensor([word2vec["love"]]).long().cuda()
like = torch.Tensor([word2vec["like"]]).long().cuda()
longer = torch.Tensor([word2vec["longer"]]).long().cuda()
long = torch.Tensor([word2vec["long"]]).long().cuda()
short = torch.Tensor([word2vec["short"]]).long().cuda()
shorter = torch.Tensor([word2vec["shorter"]]).long().cuda()

vec_nice = emb(nice)
vec_excellent = emb(excellent)
vec_bad = emb(bad)
vec_poor = emb(poor)
vec_ugly = emb(ugly)
vec_beautiful = emb(beautiful)
vec_love = emb(love)
vec_like = emb(like)
vec_longer = emb(longer)
vec_long = emb(long)
vec_short = emb(short)
vec_shorter = emb(shorter)

print("similarity(nice,excellent): ", similarity(vec_nice,vec_excellent))
print("similarity(nice,bad): ", similarity(vec_nice,vec_bad))
print("similarity(bad,poor): ", similarity(vec_bad,vec_poor))
print("similarity(nice,poor): ", similarity(vec_nice,vec_poor))
print("similarity(beautiful,nice): ", similarity(vec_beautiful,vec_nice))
print("similarity(ugly,nice): ", similarity(vec_ugly,vec_nice))
print("similarity(love,like): ", similarity(vec_love,vec_like))
print("similarity(long,longer): ", similarity(vec_long,vec_longer))
print("similarity(short,shorter): ", similarity(vec_short,vec_shorter))
print("similarity(long,short): ", similarity(vec_long,vec_short))
print("similarity(longer,shorter): ", similarity(vec_longer,vec_shorter))


print("Exporting word_emb_mat...")
word_emb_mat = model.embeddings.weight.cpu().detach().numpy()
np.savetxt("word_emb_mat.txt", word_emb_mat)

