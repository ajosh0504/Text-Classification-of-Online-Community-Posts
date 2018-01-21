"""
Created on Wed Oct 25 16:25:13 2017

@author: apoorva
"""
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords 
import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import pickle
from nltk.tokenize import RegexpTokenizer
flag=True

#Tokenization and string cleaning
vocab={}
index=0
stop = stopwords.words('english')
def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    return tokens

#Loading data and labels
def load_data_and_labels():
    input_x=[]
    label_y=[]
    dataset=pd.read_csv('re_extraction_2017_10_1.csv',encoding='latin1')
    for i in range(0,1330):
        #Data
        tmp=dataset.iloc[i]['Text']
        tmp=tmp.strip()
        tmp = preprocess(tmp)
        temp_list=[]
        for x in tmp:
                temp_word = []
                for j in x:
                    if (j >= 'a' and j <= 'z') or (j >= '0' and j <= '9'):
                        temp_word.append(j)
                temp_word = ''.join(temp_word)
                if temp_word not in stop:
                    temp_list.append(temp_word)
#Labels
        y_temp=[]
        
        for j in range(1,5):
            if flag:
                if isinstance((dataset.iloc[i]['Label'+str(j)]), basestring): #Python2
                    m=dataset.iloc[i]['Label'+str(j)] 
                
                    y_temp.append(m.strip(" ")) 
            else: 
                if isinstance((dataset.iloc[i]['Label'+str(j)]), str): #Python3
                    m=dataset.iloc[i]['Label'+str(j)] 
                
                    y_temp.append(m.strip(" ")) 
        
        over_sample_flag = False
        
        #Creating label vector corresponding to the class
        if "'com'" in y_temp:
            y = 1
            
        else:
            y = 0
        
        input_x.append(temp_list)
        label_y.append(y)
    return [input_x,label_y]

#Pads all sentences to the same length. The length is defined by the longest sentence.Returns padded sentences
def pad_sentences(sentences, padding_word=""):
    
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

# Load and preprocess data
sentences, labels = load_data_and_labels()
sentences_padded = pad_sentences(sentences)

#-------Word2Vec---------
import gensim

model = gensim.models.Word2Vec.load('model_embedding.bin')
w2v = dict(zip(model.wv.index2word, model.wv.syn0))

#Creating word embeddings- higher dimension to lower dimension
class TfidfEmbeddingVectorizer(object): #Alternative is MeanEmbeddingVectorizer, gives worse results
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idfs
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])



a = TfidfEmbeddingVectorizer(w2v)
a = a.fit(sentences_padded, labels)
a1 = a.transform(sentences_padded)
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

all_posts = np.load('all_com.npy')
tot=np.zeros((2854280))
#10 fold CV
for i in range(10):
    X_temp=list(a1)
    y_temp=list(labels)
    del X_temp[i*133:(i+1)*133]
    del y_temp[i*133:(i+1)*133]
    #Smote on the training set of each fold
    sm = SMOTE(ratio = 'auto',k_neighbors=5)
    X_temp, y_temp = sm.fit_sample(X_temp,y_temp)
    #Training the Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)

    classifier.fit(X_temp, y_temp) 
    #Predicting the Test set results
    tot= tot+classifier.predict(all_posts)
tot_arr= np.zeros((2854280))
for i in range(2854280):
    if tot[i] >= 5:
        tot_arr[i]=1
        
with open('com_res.npy','wb') as f:
    np.save(f,tot_arr)        
        
    
