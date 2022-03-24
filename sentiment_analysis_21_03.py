#!/usr/bin/env python
# coding: utf-8


#import topicnet as tn
#from topicnet.cooking_machine.config_parser import build_experiment_environment_from_yaml_config
# from topicnet.cooking_machine.pretty_output import make_notebook_pretty
# #from topicnet.cooking_machine.recipes import BaselineRecipe
# from topicnet.viewers.top_tokens_viewer import TopTokensViewer
#from gensim.corpora.dictionary import Dictionary
#from gensim.models import LdaModel

#import importlib.util
#spec = importlib.util.spec_from_file_location("text_preproc", "./Desktop/Практика/text_preproc.py")
#foo = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(foo)
#foo.MyClass()



import requests as req
import json
import urllib.parse
import pandas as pd
import subprocess
import io
import re
import ast
from bs4 import BeautifulSoup
import numpy as np
import os
import pandas as pd
import itertools
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pymorphy2 import MorphAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import spacy
from itertools import combinations
from nltk import ngrams
import nltk
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os
from natasha import NamesExtractor, MorphVocab, DatesExtractor, MoneyExtractor, AddrExtractor
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from multiprocessing import Process
from itertools import product
import time
import threading
from copy import copy
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
from collections import Counter
import math
import nltk
from sklearn.metrics import f1_score
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

from textblob import TextBlob

import text_preproc as txtprpc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#Полный препроцессинг СТРОКИ

def preproc_line(line, stop_words):
    
    line = txtprpc.rm_punctuation(line)
    
    line = txtprpc.rm_special(line)
    
    line = txtprpc.rm_numbers(line)
    
    line = txtprpc.make_lowercase(line)
    
    line = txtprpc.rm_emojies(line)
    
    line = txtprpc.rm_punctuation(line)
    
    line = txtprpc.rm_extra_symbols(line)
    
    if len(stop_words):
        
        line = txtprpc.rm_stopwords(line, stop_words)
        
    line = txtprpc.pymorphy_preproc(line)
    
    line = ' '.join(line)
    
    line = txtprpc.sub_names(line)
    
    line = txtprpc.sub_dates(line)
    
    line = txtprpc.sub_addr(line)
    
    line = txtprpc.sub_money(line)
    
    return line



#Полный препроцессинг КОРПУСА

def preproc_data(train_data, stop_words, content):
    
    corpus = [''.join(preproc_line(line, stop_words)) for line in list(train_data[content].astype(str))]
    
    #X = ident_tokenizer(corpus)

    return corpus


words_pack = txtprpc.get_stopwords_bag("stopwords_aug")

data = pd.read_excel('report_6089483914031a337d47ece6_3.xlsx', skiprows=1)

data = data.iloc[:250]


#Полный препроцессинг корпуса, обычный

start_time = time.time()

X_train = preproc_data(data, ' ', 'Текст')

print("--- %s seconds ---" % (time.time() - start_time))


data['text_proceed'] = X_train


# In[94]:

from dostoevsky.dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

#на новых данных



def get_emotions(sent):
    
    tokenizer = RegexTokenizer()

    model = FastTextSocialNetworkModel(tokenizer = tokenizer)
    
    emotions = model.predict(sent, k=1)
    
    return list(emotions.items())[0][0]


# In[94]:
    
    
start_time = time.time()

data['emotion_new'] = data['text_proceed'].apply(lambda x: get_emotions(x))

print("--- %s seconds ---" % (time.time() - start_time))

# In[94]:
    
    
start_time = time.time()

data['emotion_old'] = data['Текст'].apply(lambda x: get_emotions(x))

print("--- %s seconds ---" % (time.time() - start_time))




