import numpy as np
import io

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from tqdm import tqdm

#nltk.download('words')
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

def read_data(_file, cleaning):
    revs = []
    words_list = []
    with io.open(_file, "r",  encoding="ISO-8859-1") as f:
        next(f)
        for line in f:
            ID, label, sentence = line.split('\t')
            label_idx = 1 if label == 'pos' else 0 # 1 for pos and 0 for neg
            rev = []
            rev.append(sentence.strip())

            orig_rev = clean_str(" ".join(rev))
            
            revs.append({'y': label_idx, 'txt': orig_rev})
        
    return revs

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
    sentence = sentence.replace('\t',' ')

    return sentence

def data_preprocess(_file, cleaning, max_vocab_size):
    revs = read_data(_file, cleaning)
    return revs

def embedding_score(revs, word2vec, wrd_emb):
    """
    TODO:
        Convert sentences into vectors using BoW.
        data is a 2-D array with the size (nb_sentence*nb_vocab)
        label is a 2-D array with the size (nb_sentence*1)
    """
    data = []
    label = []
    for sent_info in revs:
        label.append([sent_info['y']])
    
        sent_wlist = sent_info['txt'].split()
        sent_vec = []
        for word in sent_wlist:
            if word not in word2vec:
                continue
            else:
                word_vec = int(word2vec[word])
                sent_vec.append(word_vec)
                
        vec_aggr = np.zeros(wrd_emb[0].shape[0])
        
        if len(sent_vec) != 0:
            for word_vec in sent_vec:
                vec_aggr += wrd_emb[word_vec]
            vec_aggr = vec_aggr / len(sent_vec)
            
        data.append(vec_aggr)
    
    return np.array(data), np.array(label)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
    parser.add_argument('-f', '--file', help='input csv file', required=False, default='./twitter-sentiment.csv')
    parser.add_argument('-c', '--clean', help='True to do data cleaning, default is False', action='store_true')
    parser.add_argument('-mv', '--max_vocab', help='max vocab size predifined, no limit if set -1', required=False, default=-1)
    args = vars(parser.parse_args())
    print(args)
    
    print("loading word_emb_mat.txt...")
    wrd_emb = np.loadtxt("word_emb_mat.txt")
    
    revs = data_preprocess(args['file'], args['clean'], int(args['max_vocab']))

    data, label = embedding_score(revs, word2vec, wrd_emb)
    
