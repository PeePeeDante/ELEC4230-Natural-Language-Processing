import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize

nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

def sentence_to_tokens(sent):

    # for return
    cleaned_tokens = []

    sent = sent.lower()
    encoded_sent = sent.encode('ASCII', 'ignore')  # encode & decode utf-8
    sent = encoded_sent.decode('ASCII', 'ignore')

    sent = re.sub("<unk>", "", sent)  # remove <unk>
    sent = re.sub("@\S+", "", sent)  # Remove mentions
    sent = re.sub("https*\S+", "", sent)  # Remove URL
    sent = re.sub("#\S+", "", sent)  # Remove hashtags
    sent = re.sub("\'\w+", "", sent)  # Remove ticks and the next character
    sent = re.sub('[%s]' % re.escape(string.punctuation), ' ', sent)
    sent = re.sub(r'\d+', '', sent)
    sent = re.sub('\s{2,}', " ", sent)  # Replace the over spaces
    
    temp = []
    lemmatizer = WordNetLemmatizer()
    for word in sent.split(" "):
        for w in word:
            if w in (string.punctuation + string.digits):
                break
        word = lemmatizer.lemmatize(word)
        temp.append(word)
        
    
    sent = ' '.join(temp)

    stop_words = stopwords.words("english")
    sent = ' '.join([word for word in sent.split(' ') if word not in stop_words])  # Remove stopwords

    return sent 

def fivegram(sent):
    sent = sent.split()
    samples = []
    if len(sent) == 1:
        return samples
    for i in range(len(sent)-1):
        if i == 3:
            break
        else:
            sample = []
            sample = [sent[x] for x in range(i+1)]
            sample.insert(0,'SOS')
            for _ in range(0,3-i):
                sample.append('PAD')
        samples.append([sample,sent[i+1]])
    if len(sent) - 5 < 1:
        return samples

    for i in range(len(sent)-5):     
       samples.append([sent[i:i+5],sent[i+5]])   
    return samples
