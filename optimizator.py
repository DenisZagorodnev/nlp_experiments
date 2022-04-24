

#ТУТ ПРИМЕР ПОИСКА НАИЛУЧШЕГО СЦЕНАРИЯ ОБРАБОТКИ И ТЕМАТИЧЕСКОГО МОДЕЛИРОВАНИЯ, МЕТРИКИ И ОПТИММИЗАЦИЯ

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import time
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from sklearn.metrics import pairwise_distances
import text_preprocessor as txtprpc
import metrices


    
#целевая функция в процессе оптимизации алгоритма тематического моделирования и процесса предобработки
   
def cluster_modeling_calc(X_train_vect, topic_words, prediction, vectorizer):
    
    #косинусные расстояния матрица
    
    for topic, words in topic_words.items():
        
        cosin_dists = []
        
        #print('Topic: %d' % topic, words)
        
        #print('Cosin sim: ', np.mean(pairwise_distances(vectorizer.transform(words).toarray(), metric='cosine')))
        
        cosin_dists.append(np.mean(pairwise_distances(vectorizer.transform(words).toarray(), metric='cosine')))
        
        
    #Среднее косинусное расстояние векторизованных tfidf слов внутри тем, что эквивалентно понятию энтропии в теории информации
        
    mean_cos_dist = np.mean(cosin_dists)
       
        
    #Среднее косинусное расстояние тем
    
    #from scipy import spatial
    
    uniques = []
    
    for i in range(len(topic_words.items())):
        
        for j in range(len(topic_words.items())):
            
            uniques.append(txtprpc.cosine_similarity(' '.join(topic_words[i]), ' '.join(topic_words[j])))
            
            #uniques.append(txtprpc.cosine_similarity(vectorizer.transform(topic_words[i]).toarray(), 
             #                                        vectorizer.transform(topic_words[j]).toarray()))
             
             #uniques.append(1 - spatial.distance.cosine(vectorizer.transform(topic_words[i]).toarray(), 
              #                                       vectorizer.transform(topic_words[j]).toarray()))
            
            
    mean_cos_dist_global = np.mean(uniques)
   
    #средний коэффициент силуэта кластеров
   
    silhouette = metrices.silhouette_avg(X_train_vect,  np.asarray(prediction))
   
    #метрика, нужная для задачи максимизации (свертка трех показателей) 
        
    result_score = 0.33*mean_cos_dist + 0.33*mean_cos_dist_global + 0.33*silhouette
   
    return result_score
    

