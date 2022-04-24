

#ТУТ ПРИМЕР ВЕКТОРИЗАЦИИ, ПРЕДОБРАБОТКИ, ОБУЧЕНИЯ КЛАСТЕРИЗАЦИИ И ДЕМОНСТРАЦИИ КЛЮЧЕВЫХ СЛОВ ПО КЛАСТЕРАМ

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import time
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from sklearn.metrics import silhouette_score, pairwise_distances
import text_preprocessor as txtprpc


#Полный препроцессинг СТРОКИ

def preproc_line(line, stop_words, extra_stopwords):
    
    #вынули смайлики, добавили к строке сущность "найден смайлик", посчитали их эмоциональный вес и удалили
    
    emojies = txtprpc.extract_emojies(line)
    
    
    line_emojies_scoring = 0
    
    for emoj in emojies:
        
        line_emojies_scoring += txtprpc.get_emojies_emotion(emoj)['sentiment_score']
        
        
    #line = txtprpc.find_out_emojies(line)
        
    line = txtprpc.rm_emojies(line)
    
    
    #вынули скобочки, добавили к строке сущность "найдены смайлики-скобочки", посчитали их эмоциональный вес и удалили
    
    #line = txtprpc.extract_hren(line)
    
    num_hren_negatives = 0 
    
    num_hren_positives = 0
    
    num_hren_negatives, num_hren_positives = txtprpc.extract_hren(line)
    
    
    #подчистили пунктуацию, числа, спец символы, значки типа \n и привели к нижнему регистру
    
    line = txtprpc.rm_punctuation(line)
    
    line = txtprpc.rm_special(line)
    
    line = txtprpc.rm_numbers(line)
    
    line = txtprpc.make_lowercase(line)
    
    line = txtprpc.rm_extra_symbols(line)
    
    #подсчет количества частей речи
    
    num_speech_tags = txtprpc.count_part_speech_tag(line)
    
    #подсчет количества используемых языков, добавили к строке сущности "найден язык NAME"
    
    num_languages = -1
    
    num_languages += txtprpc.count_languages(line)
    
    #line = txtprpc.count_languages_chardet(line)

    
    #удалили стоп-слова, если есть (до нормализации и количества частей речи)
    
    
    if len(stop_words):
        
        line = txtprpc.rm_stopwords(line, stop_words)
        
    else:
        
        line = line.split(' ')
        

    
    #удалили  ЭКСТРА стоп-слова, то есть срез n самых редких слов, если есть (до нормализации и количества частей речи)
    
    if len(extra_stopwords):
        
        line = ' '.join(line)
        
        line = txtprpc.rm_stopwords(line, extra_stopwords)

    
    #нормализация текста
        
    line = txtprpc.pymorphy_preproc(line)
    
    line = ' '.join(line)
    
    
    #вынуть именованные сущности
    
    #имена
    
    line = txtprpc.sub_names(line)
    
    #даты
    
    line = txtprpc.sub_dates(line)
    
    #адреса
    
    line = txtprpc.sub_addr(line)
    
    #валюта
    
    line = txtprpc.sub_money(line)
    
    #соединили воедино
    
    line = ''.join(line)
    
    #вернули: предобработанный текст, уровень эмоциональности смайликов, статистику по частям речи, количество языков
    
    return {'line_processed' : line, 
            'line_emojies_scoring' : line_emojies_scoring, 
            'num_speech_tags' : num_speech_tags, 
            'num_languages' : num_languages}



#Полный препроцессинг КОРПУСА

def preproc_data(train_data, stop_words, extra_stopwords, content):
    
    corpus = [preproc_line(line, stop_words, extra_stopwords) for line in list(train_data[content].astype(str))]

    return corpus

# In[94]:

#стоп-слова снаружи

stopwords = txtprpc.get_stopwords_bag("stopwords_aug")

#данные отчета

data = pd.read_excel('../../Март 2 половина.xlsx', skiprows = 1)


# In[94]:
#экспериментальная очистка от экстра стоп-слов: удалить n самых РЕДКИХ слов

#ВОПРОС: НОРМАЛИЗОВАТЬ ЛИ СТОП СЛОВА

extra_stopwords_rare = txtprpc.get_rare_n_words(data['Текст'], n = 10)


#экспериментальная очистка от экстра стоп-слов: удалить n самых ЧАСТЫХ слов

extra_stopwords_freq = txtprpc.get_freq_n_words(data['Текст'], n = 100)




# In[94]:

#Полный препроцессинг корпуса, обычный

start_time = time.time()

X_train = preproc_data(data, stopwords, extra_stopwords_freq, 'Текст')

print("--- %s seconds ---" % (time.time() - start_time))


# In[94]:

data_proceed = pd.DataFrame(X_train)   


pd.concat([data, data_proceed], axis=1).to_excel('../../Март 2 половина PROC 3.xlsx')
    
    
# In[94]:
    
    
data_proceed = pd.read_excel('../../Март 2 половина PROC 3.xlsx')


data_proceed = data_proceed[data_proceed['line_processed'].isna() == False]



# In[94]:
#векторизовали слова

vectorizer = TfidfVectorizer(ngram_range = (1, 3))

X_train_vect = vectorizer.fit_transform(data_proceed['line_processed'].dropna())


# In[94]:

#LatentDirichletAllocation алгоритм кластеризации

lda = LDA(n_components=10,random_state=17, n_jobs = -1)

#обучили на векторизованном корпусе

lda.fit(X_train_vect)

#предсказали (можно на чем угодно)

prediction = np.matrix(lda.transform(X_train_vect)).argmax(axis=1)

data_proceed['predicted_clusters'] = [row[0] for row in np.asarray(prediction)]

#число отображаемых ключевых слов кластеров

n_top_words = 10

#вынем ключевые слова и напечатаем рядом с соответствующим кластером

vocab = vectorizer.get_feature_names()

topic_words = {}

for topic, comp in enumerate(lda.components_):
 
    word_idx = np.argsort(comp)[::-1][:n_top_words]
    
    topic_words[topic] = [vocab[i] for i in word_idx]
    
for topic, words in topic_words.items():
    
    print('Topic: %d' % topic)
    
    print('  %s' % ', '.join(words))












