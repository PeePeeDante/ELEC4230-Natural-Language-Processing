import torch
import gzip
from torch.utils.data import Dataset, DataLoader

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from tqdm import tqdm

nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

BSZ = 500
WIN_SIZE = 2
MAX_WORD_SIZE = 10
MIN_WORD_SIZE = 2
FILE_NAME = "reviews_data.txt.gz"
LOW_FREQ_WORDS = 1
HIGH_FREQ_WORDS = 80

def clean_str(sentence):
    sentence = sentence.lower()
    sentence = sentence.encode('ascii', 'ignore').decode()

    temp = []
    lemmatizer = WordNetLemmatizer()
    for word in word_tokenize(sentence):
        for w in word:
            if w in (string.punctuation + string.digits):
                break
        word = lemmatizer.lemmatize(word)
        temp.append(word)
  
    sentence = ' '.join(temp)
    temp.clear()
    
    sentence = re.sub("@\S+", "", sentence) # Remove mentions
    sentence = re.sub("https*\S+", "", sentence) # Remove URL
    sentence = re.sub("#\S+", "", sentence) # Remove hashtags
    sentence = re.sub("\'\w+", "", sentence) # Remove ticks and the next character
    sentence = re.sub('[%s]' % re.escape(string.punctuation), "", sentence)
    sentence = re.sub(r'\w*\d+\w*', '', sentence) # Remove numbers
    sentence = re.sub('\s{2,}', " ", sentence) # Replace the over spaces
    
    stop_words = stopwords.words("english")
    sentence = ' '.join([word for word in sentence.split(' ') if word not in stop_words]) # Remove stopwords

    return sentence

class MyData(Dataset):
    def build_vocab(self):
        pass
        
    def __init__(self, filename = FILE_NAME):
        self.len = 0
        print("reading byte file...")
        with gzip.open(filename, 'rb') as f:
            original_text = [x.strip() for x in f if x.strip()]

        cleaned_text = []
        print("cleaning original text...")
        for sentence in tqdm(original_text):
            sentence = sentence.lower().decode('Latin1')
            comment_start = sentence.find('\t')+1
            sentence = clean_str(sentence[comment_start:]).strip()
            sentence = sentence.replace('\t',' ')

            # cleaning words in sentences that has extreme length
            temp = sentence.split()
            word_list = [w for w in temp if len(w) <= MAX_WORD_SIZE and len(w) >= MIN_WORD_SIZE]
            temp.clear() # save RAM
            sentence = ' '.join(word_list)

            if sentence != '':
                cleaned_text.append(sentence)

        original_text.clear() # save RAM

        self.num_sent = len(cleaned_text)
        self.word2vec, self.vocab_count = self.build_vocab(cleaned_text)

        print(self.word2vec)
        
        print("converting sentences to vectors...")
        sents_vectors = []
        for sent in tqdm(cleaned_text):
            word_list = sent.split()

            sent2vec = []
            for word in word_list:
                # remove 0's (i.e. "UNK")
                if word not in self.word2vec:
                    continue
                else:
                    sent2vec.append(self.word2vec[word])

            if len(sent2vec) < 5:
                continue

            sents_vectors.append(sent2vec)
        
        cleaned_text.clear() # save RAM

        self.target_context = []
        print("building target_context tuple...")
        for sent2vec in tqdm(sents_vectors):
            for i in range(WIN_SIZE,len(sent2vec)-WIN_SIZE):
                target = sent2vec[i]
                context = []
                for j in range(WIN_SIZE):
                    context.append(sent2vec[i+j-WIN_SIZE])
                for j in range(WIN_SIZE):
                    context.append(sent2vec[i+j+1])
                self.target_context.append( [target, context] )

        self.len = len(self.target_context)
            
    def build_vocab(self, cleaned_text):

        print("building vocab class...")
        freq = {"UNK": 0}
        for sentence in tqdm(cleaned_text):
            word_list = sentence.split()

            if len(word_list)<5:
                continue
                
            for word in word_list:
                if word not in freq:
                    freq[word] = 1
                else:
                    freq[word] += 1
        
        count = {0:0}
        for word in freq:
            if freq[word] not in count:
                count[freq[word]] = 1
            else:
                count[freq[word]] += 1

        print("statistics of vocab frequency")
        stats = {k: v for k, v in sorted(count.items(), key=lambda item: item[1])}
        print(stats)

        delete = [word for word in freq if (freq[word] >= LOW_FREQ_WORDS and freq[word] <= HIGH_FREQ_WORDS)]
        for word in delete:
            del freq[word]
            freq["UNK"] += 1

        idx = 1
        word2vec = {'UNK': 0}
        for word in freq:
            if word == "UNK":
                continue
            word2vec[word] = idx
            idx += 1
        
        print("word2vec size: ", len(word2vec))
        return word2vec, len(word2vec)

    def get_word2vec(vocab):
        return self.word2vec[vocab]
    
    def __getitem__(self, index):
        return self.target_context[index][0], self.target_context[index][1]

    def __len__(self):
        return self.len
