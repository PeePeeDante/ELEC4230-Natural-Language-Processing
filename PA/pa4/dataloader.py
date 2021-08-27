import operator
from pickle import FALSE
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import sentence_to_tokens, fivegram

PAD_INDEX = 0
UNK_INDEX = 1
SOS_INDEX = 2
TRAIN_FILE = "train.txt"
DEV_FILE = "valid.txt"

class Vocab():
    def __init__(self):
        self.word2index = {"PAD": PAD_INDEX, "UNK": UNK_INDEX,"SOS": SOS_INDEX}
        self.word2count = {}
        self.index2word = {PAD_INDEX: "PAD", UNK_INDEX: "UNK", SOS_INDEX: "SOS"}
        self.n_words = 3  # Count default tokens
        self.word_num = 0
        self.sen_num = 0

    def index_words(self, sentence):
        self.sen_num += 1
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
        self.n_words = 3
        new_word2index = {"PAD": PAD_INDEX, "UNK": UNK_INDEX, "SOS": SOS_INDEX}
        new_index2word = {PAD_INDEX: "PAD", UNK_INDEX: "UNK", SOS_INDEX: "SOS"}
        for gd_wd in self.word2index:
            if (gd_wd != "PAD" and gd_wd != "UNK" and gd_wd != "SOS"):
                new_word2index[gd_wd] = self.n_words
                new_index2word[self.n_words] = gd_wd
                self.n_words += 1

        self.word2index = new_word2index
        self.index2word = new_index2word

def build_vocab(vocab,filename):
    statistic = {"sent_num": 0, "word_num": 0, "vocab_size": 0,
             "max_len": 0, "avg_len": 0}

    # read dataset
    with open(filename, 'r', encoding="UTF-8") as f:

        # read each paragraph
        for _, paragraph in tqdm(list(enumerate(f))):

            sentences = paragraph.split('.')  # split to sentences
            statistic["sent_num"] += len(sentences)  # count sentence number

            for i in range(len(sentences)):
                sentences[i] = sentence_to_tokens(sentences[i])

                # find max length of sentence (after cleaning)
                sent_len = len(sentences[i].split())
                statistic["word_num"] += sent_len  # count word number
                if sent_len > statistic["max_len"]:
                    statistic["max_len"] = sent_len

                # build vocab
                vocab.index_words(sentences[i])

    # clean low frequency words
    vocab.clean_bad_words()
    statistic["avg_len"] = statistic["word_num"]/statistic["sent_num"]
    statistic["word_num"] = vocab.word_num
    statistic["vocab_size"] = vocab.n_words

    return vocab, statistic

class MyData (Dataset):
    def __init__(self, filename, vocab):

        # class variable
        self.samples = []
        self.sample_num = 0
        self.vocab = vocab

        print('convert '+ filename + ' to samples')
        # read dataset
        with open(filename, 'r', encoding = "UTF-8") as f:

            # read each paragraph
            for _ , paragraph in tqdm(list(enumerate(f))):
                sentences = []
                sen_samples = []
                
                sentences = paragraph.split('.')  # split to sentences

                # preprocess each sentence
                for i in range(len(sentences)):
                    sentences[i] = sentence_to_tokens(sentences[i])
                    
                    # convert sentence to samples
                    sen_samples = fivegram(sentences[i])

                    # append samples to sample set
                    for j in range(len(sen_samples)):
                        self.samples.append(sen_samples[j])
                        
        self.sample_num = len(self.samples)

    def info(self):
        print('sample number:', self.sample_num)
        print('sentence number:', self.vocab.sen_num)
        print('vocab number:', self.vocab.n_words)
        print('word number:', self.vocab.word_num)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):

        # X
        X_return = []
        for i in range(5):
            if self.samples[idx][0][i] in self.vocab.word2index.keys():
                X_return.append(self.vocab.word2index[self.samples[idx][0][i]])
            else:
                X_return.append(self.vocab.word2index['UNK'])

        # Y
        if self.samples[idx][1] in self.vocab.word2index.keys():
            Y_return = self.vocab.word2index[self.samples[idx][1]]
        else:
            Y_return = self.vocab.word2index['UNK']
        return torch.LongTensor(X_return), Y_return

def get_dataloaders(batch_size):
    vocab = Vocab()
    vocab, statistic = build_vocab(vocab, TRAIN_FILE)

    train_data = MyData(TRAIN_FILE,vocab)
    dev_data = MyData(DEV_FILE,vocab)

    train_data.info()

    train_loader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle = False)
    dev_loader = DataLoader(dataset=dev_data, batch_size = batch_size, shuffle = False)

    return train_loader, dev_loader, train_data.vocab.n_words

# Test the loader
if __name__ == "__main__":
    get_dataloaders(batch_size=32)
