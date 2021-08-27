import pandas as pd
import re
import numpy as np
import pickle

import torch
import torch.utils.data as data
import string

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from tqdm import tqdm

nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')


PAD_INDEX = 0
UNK_INDEX = 1
def clean(sent):
    # Clean the data
    ############################################################
    # TODO
    ############################################################
    sent = sent.lower()
    sent = re.sub("@\S+", "", sent) # Remove mentions
    sent = re.sub("https*\S+", "", sent) # Remove URL
    sent = re.sub("#\S+", "", sent) # Remove hashtags
    sent = re.sub("\'\w+", "", sent) # Remove ticks and the next character
    sent = re.sub('[%s]' % re.escape(string.punctuation), ' ', sent)
    sent = re.sub(r'\d+', '', sent)
    sent = re.sub('\s{2,}', " ", sent) # Replace the over spaces

    temp = []
    lemmatizer = WordNetLemmatizer()
    #for word in word_tokenize(sent):
    for word in sent.split(" "):
        for w in word:
            if w in (string.punctuation + string.digits):
                break
        word = lemmatizer.lemmatize(word)
        temp.append(word)
    sent = ' '.join(temp)
    temp.clear()

    stop_words = stopwords.words("english")
    sent = ' '.join([word for word in sent.split(' ') if word not in stop_words]) # Remove stopwords

    return sent
'''
def clean(sent):
    # Clean the data
    ############################################################
    # TODO
    ############################################################
    sent = sent.lower()
    sent = sent.encode('ascii', 'ignore').decode()

    temp = []
    lemmatizer = WordNetLemmatizer()
    for word in word_tokenize(sent):
        for w in word:
            if w in (string.punctuation + string.digits):
                break
        word = lemmatizer.lemmatize(word)
        temp.append(word)
    sent = ' '.join(temp)
    temp.clear()
    
    sent = re.sub("@\S+", "", sent) # Remove mentions
    sent = re.sub("https*\S+", "", sent) # Remove URL
    sent = re.sub("#\S+", "", sent) # Remove hashtags
    sent = re.sub("\'\w+", "", sent) # Remove ticks and the next character
    sent = re.sub('[%s]' % re.escape(string.punctuation), "", sent)
    sent = re.sub(r'\w*\d+\w*', '', sent) # Remove numbers
    sent = re.sub('\s{2,}', " ", sent) # Replace the over spaces
    
    stop_words = stopwords.words("english")
    sent = ' '.join([word for word in sent.split(' ') if word not in stop_words]) # Remove stopwords

    return sent
    '''

class Vocab():
    def __init__(self):
        self.word2index = {"PAD": PAD_INDEX, "UNK": UNK_INDEX}
        self.word2count = {}
        self.index2word = {PAD_INDEX: "PAD", UNK_INDEX: "UNK"}
        self.n_words = 2 # Count default tokens
        self.word_num = 0

    def index_words(self, sentence):
        for word in sentence.split():
            self.word_num += 1
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words += 1
            else:
                self.word2count[word] += 1
    
    def clean_bad_words(self):
        bad_words = []
        for k in self.word2count:
            if (self.word2count[k] == 1):
                bad_words.append(k)
        
        # reserve only good_words
        for bd_wd in bad_words:
            del self.word2index[bd_wd]
            del self.word2count[bd_wd]
        
        # reassign index to self.word2index, self.index2word, self.n_words
        # self.word2count doesn't require update
        self.n_words = 2
        new_word2index = {"PAD": PAD_INDEX, "UNK": UNK_INDEX}
        new_index2word = {PAD_INDEX: "PAD", UNK_INDEX: "UNK"}
        for gd_wd in self.word2index:
            if (gd_wd!= "PAD" and gd_wd!= "UNK"):
                new_word2index[gd_wd] = self.n_words
                new_index2word[self.n_words] = gd_wd
                self.n_words += 1
        
        self.word2index = new_word2index
        self.index2word = new_index2word
        

def Lang(vocab, file_name):
    statistic = {"sent_num": 0, "word_num": 0, "vocab_size": 0, "max_len": 0, "avg_len": 0, "len_std": 0, "class_distribution": {}}
    df = pd.read_csv(file_name)
    statistic["sent_num"] = len(df)
    sent_len_list = []
    ############################################################
    # TO DO
    # Build vocabulary and statistic
    
    statistic["class_distribution"] = {'0':0,'1':0,'2':0}
    wd_idx = 2
    for i in range(len(df)):
        sent = clean(str(df['content'][i]).strip())
        sent_len_list.append(len(sent.split()))
        
        vocab.index_words(sent)
        
        statistic["class_distribution"][str(df['rating'][i])] += 1
    
    vocab.clean_bad_words()
    
    statistic["word_num"] = vocab.word_num
    statistic["vocab_size"] = vocab.n_words
    # print(vocab.word2index) # print dictionary
    statistic["max_len"] = max(sent_len_list)
    statistic["avg_len"] = sum(sent_len_list) / len(sent_len_list)
    
    s = 0
    for i in range(len(sent_len_list)):
        s += (sent_len_list[i] - statistic["avg_len"])**2
    
    var = s / (len(sent_len_list)-1)
    statistic["len_std"] = var**(1/2)
    
    ############################################################
    return vocab, statistic


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab):
        self.id, self.X, self.y = data
        self.vocab = vocab
        self.num_total_seqs = len(self.X)
        self.id = torch.LongTensor(self.id)
        if self.y is not None:
            self.y = torch.LongTensor(self.y)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ind = self.id[index]
        X = self.tokenize(self.X[index])
        if(self.y is not None):
            y = self.y[index]
            return torch.LongTensor(X), y, ind
        else:
            return torch.LongTensor(X), ind

    def __len__(self):
        return self.num_total_seqs

    def tokenize(self, sentence):
        return [self.vocab.word2index[word] if word in self.vocab.word2index else UNK_INDEX for word in sentence]

def preprocess(filename, max_len=200, test=False):
    df = pd.read_csv(filename)
    id_ = [] # review id
    rating = [] # rating
    content = [] # review content

    for i in range(len(df)):
        id_.append(int(df['id'][i]))
        if not test:
            rating.append(int(df['rating'][i]))
        sentence = clean(str(df['content'][i]).strip())
        sentence = sentence.split()
        sent_len = len(sentence)
        # Here we pad the sequence for whole training set, you can also try to do dynamic padding for each batch by customize collate_fn function
        # If you do dynamic padding and report it, we will give 1 points bonus
        if sent_len > max_len:
            content.append(sentence[:max_len]) # truncate the excessive part
        else:
            content.append(sentence + ["PAD"] * (max_len - sent_len))

    if test:
        len(id_) == len(content)
        return (id_, content, None)
    else:
        assert len(id_) == len(content) == len(rating)
        return (id_, content, rating)

def get_dataloaders(batch_size, max_len):
    vocab = Vocab()
    vocab, statistic = Lang(vocab, "train.csv")

    train_data = preprocess("train.csv", max_len)
    dev_data = preprocess("dev.csv", max_len)
    test_data = preprocess("test.csv", max_len, test=True)
    train = Dataset(train_data, vocab)
    dev = Dataset(dev_data, vocab)
    test = Dataset(test_data, vocab)
    print(statistic)
    data_loader_tr = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    data_loader_dev = torch.utils.data.DataLoader(dataset=dev, batch_size=batch_size, shuffle=False)
    data_loader_test = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
    return data_loader_tr, data_loader_dev, data_loader_test, statistic["vocab_size"]
