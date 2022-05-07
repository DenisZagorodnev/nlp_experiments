

#ТУТ ПРИМЕР ПОИСКА НАИЛУЧШЕГО СЦЕНАРИЯ ОБРАБОТКИ И ТЕМАТИЧЕСКОГО МОДЕЛИРОВАНИЯ, МЕТРИКИ И ОПТИММИЗАЦИЯ

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import time
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
import text_preprocessor as txtprpc
from optimizator import cluster_modeling_calc
import copy


#стоп-слова снаружи

stopwords = txtprpc.get_stopwords_bag("stopwords_aug")

#важные колонки: 'ID поста', 'Текст', 'Эмоциональный окрас', 'Все посты', 'Все комментарии'

data_example = pd.read_excel('./data/Пример.xlsx', 'Все комментарии', skiprows=1)

data_example['Текст'] = data_example['Текст'].astype(str)

#пост и число его комментариев

post_sizes = dict(data_example.groupby(['ID поста']).size())

post_sizes_sorted = {k: v for k, v in sorted(post_sizes.items(), key=lambda item: -item[1])}

#экстра стоп-слова: самые популярные слова корпуса с учетом оценки длины

extra_stopwords = txtprpc.get_freq_n_words(data_example['Текст'].astype(str), n = 50)



# In[94]:

#склейщик списка слов в предложение

def concatter(line):
    
    return  ' '.join(line)


#разделитель предложения на слова по пробелам

def splitter(line):
    
    return line.split(' ')
    
#функция предобработки отедльного элемента корпуса: удаление 
#стоп-слов и применение последовательности функций предобработки

def preproc_line(line, stop_words, extra_stopwords, funcs_seq):
    
    for func in funcs_seq:
        
        try:
        
            line = func(line)
            
        except:
            
            None
        
    return line
    
#удалитель стоп-слов

def rm_stopwords(line):
            
    return txtprpc.rm_stopwords(line, stopwords) 

#удалитель экстра стоп-слов

def rm_extra_stopwords(line):
            
    return txtprpc.rm_stopwords(line, extra_stopwords)

#последовательность функций, формирующих предобработку
#в этой версии есть все функции


funcs_seq_0 = [#txtprpc.find_out_emojies,
               txtprpc.extract_hren,
               txtprpc.rm_emojies, 
               txtprpc.rm_punctuation, 
               txtprpc.rm_special, 
               txtprpc.rm_numbers, 
               txtprpc.make_lowercase, 
               txtprpc.rm_extra_symbols,
               #txtprpc.count_languages_chardet,
               rm_stopwords, 
               concatter, 
               rm_extra_stopwords, 
               txtprpc.pymorphy_preproc, 
               concatter
               #txtprpc.sub_names, 
               #txtprpc.sub_dates, 
               #txtprpc.sub_addr, 
               #txtprpc.sub_money, 
               #''.join
               ]

funcs_seq_1 = [#txtprpc.find_out_emojies,
               txtprpc.extract_hren,
               txtprpc.rm_emojies, 
               txtprpc.rm_punctuation, 
               txtprpc.rm_special, 
               txtprpc.rm_numbers, 
               txtprpc.make_lowercase, 
               txtprpc.rm_extra_symbols,
              # txtprpc.count_languages_chardet,
               rm_stopwords, 
               concatter, 
               rm_extra_stopwords, 
               txtprpc.pymorphy_preproc, 
               concatter
              # txtprpc.sub_names, 
               #txtprpc.sub_dates, 
              # txtprpc.sub_addr, 
               #txtprpc.sub_money, 
               #''.join
               ]

funcs_seq_2 = [txtprpc.rm_emojies, txtprpc.rm_punctuation, txtprpc.rm_special, 
               txtprpc.rm_numbers, txtprpc.make_lowercase, txtprpc.rm_extra_symbols, 
               rm_stopwords, 
               concatter, 
               rm_extra_stopwords, 
               splitter,  
               txtprpc.pymorphy_preproc, 
               concatter, txtprpc.sub_names, txtprpc.sub_dates, txtprpc.sub_addr, txtprpc.sub_money, ''.join]


funcs_seq_3 = [txtprpc.rm_emojies, txtprpc.rm_punctuation, txtprpc.rm_special, 
               txtprpc.rm_numbers, txtprpc.make_lowercase, txtprpc.rm_extra_symbols, 
               splitter,  
               txtprpc.pymorphy_preproc, 
               concatter,  ''.join]


funcs_seq_4 = [txtprpc.find_out_emojies, txtprpc.rm_emojies, txtprpc.extract_hren, txtprpc.rm_punctuation, txtprpc.rm_special, 
               txtprpc.rm_numbers, txtprpc.make_lowercase, txtprpc.rm_extra_symbols, txtprpc.count_languages_chardet,
               splitter,  
               txtprpc.pymorphy_preproc, 
               concatter,  ''.join]




funcs_seqs = [funcs_seq_0, funcs_seq_1 , funcs_seq_2, funcs_seq_2]


#полный препроцессинг корпуса для оптимизации с учетом стоп-слов и последовательности функций

def preproc_data(train_data, stop_words, extra_stopwords, content, funcs_seq):
    
    corpus = [preproc_line(line, stop_words, extra_stopwords, funcs_seq) for line in list(train_data[content].astype(str))]

    return corpus




# In[94]:

#предобработали корпус всеми предложенными стратегиями (последовательностями функций)

X_trains = []

start_time = time.time()

for funcs_seq in funcs_seqs:

    X_trains.append(preproc_data(data_example, stopwords, extra_stopwords, 'Текст', funcs_seq))

print("--- %s seconds ---" % (time.time() - start_time))

    
    
# In[94]:
    
#оптимизация (поиск метода предобработки и метода кластеризации, 
#которым соответствует высшее значение целевого показателя)

#проходим для всех вариантов предобработанных текстов

import multiprocessing
import torch

results = {'Data №' : 0,
                'num_topics' : 0, 
                'metric_value' : 0}

def func(X_trains, i):
    
    data = copy.copy(X_trains[i])
    
    #для различного количества тем (кластеров)
    
    for num_topics in [3, 6, 9, 12]:
        
        #векторизовали обработанный корпус с выделением n-грамм
    
        #vectorizer = TfidfVectorizer(ngram_range = (1, 2))
        
        vectorizer = TfidfVectorizer(ngram_range = (1, 2))
        
        X_train_vect = vectorizer.fit_transform(data)
        
        #алгоритм кластеризации LatentDirichletAllocation
        
        lda = LDA(n_components=num_topics,random_state=17, n_jobs = -1)
        
        lda.fit(X_train_vect)
        
        #предсказали соотвествие элементов корпуса кластерам
        
        prediction = np.matrix(lda.transform(X_train_vect)).argmax(axis=1)
        
        #новый столбец желмент-кластер
        #data_proceed['predicted_clusters'] = [row[0] for row in np.asarray(prediction)]
        
        #взяли топ ключевых 5 слов, образующих кластеры
        
        n_top_words = 5
        
        vocab = vectorizer.get_feature_names()
        
        topic_words = {}
        
        for topic, comp in enumerate(lda.components_):
         
            word_idx = np.argsort(comp)[::-1][:n_top_words]
            
            topic_words[topic] = [vocab[i] for i in word_idx]
            
        metric_val = cluster_modeling_calc(X_train_vect, topic_words, [row[0] for row in np.asarray(prediction)], vectorizer)
        
        
        #if metric_val > results['metric_value']:
            
         #   results = {'Data №' : i,
          #      'num_topics' : num_topics, 
          #      'metric_value' : metric_val}
        
        print('Data №', i,'num_topics is ', num_topics, 'metric is ', metric_val)
        
# In[94]:
        
start_time = time.time()


if __name__ == '__main__' :
    
    #multiprocessing.freeze_support()
    
    procs = []
    
    for i in range(4):
        
            p = multiprocessing.Process(target=func, args=(X_trains, i))
            
            procs.append(p)
            
            p.start()
            
    for proc in procs:
        
        proc.join()

print("--- %s seconds ---" % (time.time() - start_time))

# In[94]:







#применяем полученную на предыдущем шаге лучшую модель на 1 самом популярном посте

data_example['X_train'] = X_trains[results['Data №']]

sub_data_example = copy.copy(data_example[data_example['ID поста'] == list(post_sizes_sorted.keys())[0]])

X_train = sub_data_example['X_train']

#start_time = time.time()

#векторизовали слова

vectorizer = TfidfVectorizer(ngram_range = (1, 2))

#vectorizer = TfidfVectorizer()

X_train_vect = vectorizer.fit_transform(X_train)

lda = LDA(n_components = results['num_topics'],random_state=17, n_jobs = -1)

#обучили на векторизованном корпусе

lda.fit(X_train_vect)

#предсказали (можно на чем угодно)

prediction = np.matrix(lda.transform(X_train_vect)).argmax(axis=1)

sub_data_example['predicted_clusters'] = [row[0] for row in np.asarray(prediction)]

#число отображаемых ключевых слов кластеров

n_top_words = 20

#вынем ключевые слова и напечатаем рядом с соответствующим кластером

vocab = vectorizer.get_feature_names()

topic_words = {}

for topic, comp in enumerate(lda.components_):
 
    word_idx = np.argsort(comp)[::-1][:n_top_words]
    
    topic_words[topic] = [vocab[i] for i in word_idx]
    
for topic, words in topic_words.items():
    
    print('Topic: %d' % topic)
    
    print('  %s' % ', '.join(words))







# In[94]:
    
    
#эмоциональный окрас топ ключевых слов самого объемного поста
k_features = {}


for elem in list(topic_words.values()):
    
    for feature in elem:
        
        k_features[feature]  = {'Негатив' : 0, 'Нейтральность' : 0, 'Позитив' : 0, 'Юмор' : 0}
        

        
#{'Негатив', 'Нейтральность', 'Позитив', 'Юмор'}

for i in range(len(sub_data_example)):
    
    cl = sub_data_example.iloc[i]['predicted_clusters']
    
    for elem in list(topic_words.values())[cl]:
        
        if elem in list(sub_data_example['X_train'])[i]:
        
            k_features[elem][sub_data_example.iloc[i]['Эмоциональный окрас']] += 1

words = []

ranks = []

amounts = []           
            
    
for k, v in k_features.items():
    
    words.append(k)
    
    ranks.append(max(v, key=v.get))
    
    amounts.append(sum(v.values()))
    
    #print('Word', '\'' + k + '\'', ' has emotion rank ', '\'' + max(v, key=v.get) + '\'', ' with amount: ', sum(v.values()))
    
#sub_data_example['Эмоциональный окрас']

result = pd.DataFrame(list(zip(words, ranks, amounts)),
              columns=['Ключевые слова','Окрас контекста', 'Частота употребления']).sort_values('Окрас контекста')


result.to_excel('ключевые_слова_частоты_1.xlsx')
    
    
    
    
    
    
    
    
    
    
    

