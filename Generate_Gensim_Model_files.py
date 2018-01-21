"""
Created on Wed Oct 25 16:25:13 2017

@author: apoorva
"""

from nltk.corpus import stopwords
import pandas as pd
import pickle
from nltk.tokenize import RegexpTokenizer
flag=True
import nltk

#nltk.download('stopwords')
#Tokenization and string cleaning
index=0
stop = stopwords.words('english')
def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    return tokens
#Loading data and labels
sentence = []
def load_data_and_labels():
    input_x=[]
    dataset=pd.read_csv('posts_12_4.txt', sep= '\t', header= None, encoding='latin1')

#Structure of the data: Post ID| Comment ID| Post| Timestamp    
    for i in range(0, 2854245):
        #Data
        print(i)
        tmp=dataset.iloc[i][4]
        #print(tmp)
        tmp_list = []
        if isinstance(tmp, str):
            #tmp = clean_str(tmp)
            print(i)
            tmp= preprocess(tmp)
            #tmp= tmp.split(" ")
            for x in tmp:
                temp_word = []
                for j in x:
                    if (j >= 'a' and j <= 'z') or (j >= '0' and j <= '9'):
                        temp_word.append(j)
                temp_word = ''.join(temp_word)
                tmp_list.append(temp_word)
        if len(tmp_list):
            sentence.append(tmp_list)
load_data_and_labels()

#Building model for word embedding using Gensim
import gensim
model = gensim.models.Word2Vec(sentence, size = 300)
model.save('model_embedding.bin')
