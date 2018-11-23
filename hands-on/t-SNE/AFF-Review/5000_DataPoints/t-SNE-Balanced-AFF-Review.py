
# coding: utf-8

# # t-SNE Visualization on Amazon Food Review Dataset

# ## Import Required Modules

# In[1]:


import os # for file management
import shutil # for file management
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time # for time measurement
import imageio # for GIF creation

from sklearn.feature_extraction.text import CountVectorizer # for Bag Of Words
from sklearn.feature_extraction.text import TfidfVectorizer # for text to vector creation
from gensim.models import Word2Vec

from sklearn.preprocessing import StandardScaler # for Column Standardization - DO WE NEED THIS?
from sklearn.manifold import TSNE # for t-SNE


# In[2]:


## Configure Matplotlib for nice image in PDF
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.figsize'] = 10,6
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8


# In[3]:


output_dir = 'Output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# ## Load Data

# In[4]:


con = sqlite3.connect('./cleaned.sqlite')

df = pd.read_sql_query(""" SELECT * from Reviews""", con)
df.head()


# In[ ]:


df.describe()


# In[ ]:


df.dtypes


# In[ ]:


# Split data
# positive review score, negative review score and review text as seperate dataframes
df_text = df['CleanedText']
print(df_text.shape)
df_text.head()


# In[ ]:


df_pos = df[df['Score'] == 1]
df_neg = df[df['Score'] == 0]


# In[ ]:


df_pos.describe()


# In[ ]:


df_neg.describe()


# In[ ]:


def genTSNEGif(std_data, ndp, p, itr_list, file_prefix, closePlt=False):
    '''
    Fuction which genrate t-SNE visualtion for each itr_list using given ndp and p
    Generates a GIF and stores it under '{img_name}.gif'
    Where:
        std_data - Column Standardized Data
        ndp - Number of Data Points to consider in std_data
        p - Perplexity
        itr_list - List of iterations, each iteration will be a frame in GIF
        file_prefix - Prefix to the name of GIF image
        closePlt - If you do not want to display the generated image in Notebook
    '''
    image_name = '{0}_tsne_ndp_{1}_p_{2}.gif'.format(file_prefix,ndp,p)
    print('No.Of Data Points - {0}, Perplexity - {1}, Iterations - {2}, ImageName - {3}'.format(
            ndp, p, itr_list, image_name))
    
    # list to hold the frames
    frames = []
    #print(type(std_data), std_data.shape)
    p_data = std_data[0:ndp,:]
    p_labels = final_reviews_scores[0:ndp]
    
    #print('t-SNE Data Points {0} and its Labels {1}'.format(p_data.shape, p_labels.shape))
    for itr_val in itr_list:
        img_title = '{0}-ndp={1} p={2} itr={3}'.format(file_prefix, ndp, p, itr_val)
        
        time_start = time.time()
        
        model = TSNE(n_components=2,random_state=0,perplexity=p,n_iter=itr_val,init='pca') #,verbose=2
        tsne_data = model.fit_transform(p_data)
        time_elapsed = time.time() - time_start
        print('{0} ==> t-SNE done! Time elapsed: {1} seconds'.format(img_title, time.time() - time_start))
        
        #print('Shape of tsne_data: ',tsne_data.shape)
        #print('Shape of p_labels: ',p_labels.shape)
    
        tsne_data = np.vstack((tsne_data.T,p_labels)).T
        #print(tsne_data.shape)
        #tsne_data[:4]
        tsne_df = pd.DataFrame(tsne_data,columns=['Dim_1','Dim_2','Score'])
        #tsne_df.head()
        g = sns.FacetGrid(tsne_df,hue='Score',height=10).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend();
        g.fig.suptitle(img_title);
        g.fig.canvas.draw();
        image = np.frombuffer(g.fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(g.fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        
        if (closePlt == True):
            plt.close()
    
    kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    imageio.mimsave(Path.cwd() / output_dir / image_name, frames, fps=1)
    
    return


# ## Training Data for Visualization - 5K Points

# In[ ]:


# we can't process all 364K revies, selecting a subset of it
total_data_set_size = 5000
training_data_set_size = 5000
end_iteration = 61

# Create a Balanced dataset having both +ive and -ive reviews
df_positive_reviews = df[df.Score == 1].sample(int(total_data_set_size/2))
df_negative_reviews = df[df.Score == 0].sample(int(total_data_set_size/2))

final_reviews = pd.concat([df_positive_reviews, df_negative_reviews])
final_reviews_scores = final_reviews['Score']

print('Shape of Training Data {0}'.format(final_reviews.shape))
print('Shape of Training Label {0}'.format(final_reviews_scores.shape))


# In[ ]:


final_reviews.head()


# # Bag of Words (BoW)

# In[ ]:


'''
# Create Vectors
count_vect = CountVectorizer(ngram_range=(1,2)) # create an instance
final_counts = count_vect.fit_transform(final_reviews['CleanedText'].values)
print('Shape of BoW Vectorizer: ', final_counts.get_shape())
print('Total no.of unique words: ', final_counts.get_shape()[1])

# Standardize the Data
standardized_data = StandardScaler().fit_transform(final_counts.toarray().astype(np.float64)) #, with_mean=False
print('Shape of Standardized data', standardized_data.shape)


# In[ ]:


# On Standardized data
genTSNEGif(standardized_data, training_data_set_size, 30, range(1000,3001,1000), 'BoW-std',closePlt=True) 


# In[ ]:


# on Non-Standardization
dense_mat = final_counts.toarray().astype(np.float64)
for p in range(10, end_iteration, 10):
    genTSNEGif(dense_mat, len(dense_mat), p, range(1000,5001,1000), 'BoW',closePlt=True)
'''

# # Word2Vec
# 
# I am creating vectors having 50 dimensions.
# Just a random value, not inherent calculation I made on this size decision.

# In[ ]:


# Create List arry for creating own W2V
list_of_sent = []
for sent in final_reviews['CleanedText'].values:
    list_of_sent.append(sent.decode("utf-8").split())
    
print(final_reviews.CleanedText.values[0])
print(len(list_of_sent), list_of_sent[0])


# In[ ]:


# Required dimension
w2v_d = 50

# Considering words that are occured atleast 5 times in the corpus
w2v_model = Word2Vec(list_of_sent, min_count=5, size=w2v_d, workers=4)

w2v_words = list(w2v_model.wv.vocab)
print("number of words that occured minimum 5 times : ",len(w2v_words))
print("sample words ", w2v_words[0:50])


# ## Avg-W2V

# In[ ]:


# Computing average w2v for each review in selected training dataset
review_vectors = []
for sent in tqdm(list_of_sent, ascii=True):
    sent_vec = np.zeros(w2v_d) # array to hold the vectors. Initially assuming no vectors in this review
    no_of_words_in_review = 0 # number of words with valid vector in this review
    
    # count all the words (that are in w2v model) and take average
    for word in sent:
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            no_of_words_in_review += 1
    if no_of_words_in_review != 0:
        sent_vec /= no_of_words_in_review
    review_vectors.append(sent_vec)
    
print(len(review_vectors))
print(len(review_vectors[0]))
review_vectors = np.array(review_vectors)


# In[ ]:


# t-SNE using Average Word2Vec

#genTSNEGif(review_vectors, training_data_set_size, 30, range(1000,10001,1000), 'avg-w2v')


# In[ ]:


for p in range(10, end_iteration, 10):
    genTSNEGif(review_vectors, len(review_vectors), p, range(1000,10001,1000), 'avg-w2v',closePlt=True)


# ## TFIDF Weighted W2V
# 
# Computing tfidf weighted w2v over the selected training dataset

# In[ ]:


# Create tf-idf vector matrix
tf_idf_model = TfidfVectorizer(ngram_range=(1,2))
tf_idf_matrix = tf_idf_model.fit_transform(final_reviews['CleanedText'].values)

# Create dictionary having words (features) as keys, its tf-idf values as values
tf_idf_dict = dict(zip(tf_idf_model.get_feature_names(), list(tf_idf_model.idf_)))
len(tf_idf_dict)


# In[ ]:


tf_idf_feat = tf_idf_model.get_feature_names()

# Computing tf-idf weighted w2v for each review in selected training dataset
review_vectors = []
for sent in tqdm(list_of_sent, ascii=True):
    sent_vec = np.zeros(w2v_d) # array to hold the vectors
    no_of_words_in_review = 0 # number of words with valid vector in this review
    
    # count all the words (that are in w2v model) and take average
    for word in sent:
        if word in w2v_words:
            vec = w2v_model.wv[word]
            # calculate tf-idf weighted w2v value for this word
            tf_idf = tf_idf_dict[word] * (sent.count(word)/len(sent))
            sent_vec += (vec * tf_idf)
            no_of_words_in_review += 1
    if no_of_words_in_review != 0:
        sent_vec /= no_of_words_in_review
    review_vectors.append(sent_vec)
    
print(len(review_vectors))
print(len(review_vectors[0]))
review_vectors = np.array(review_vectors)


# In[ ]:


# t-SNE using tf-idf weighted s2v

#genTSNEGif(review_vectors, training_data_set_size, 30, range(1000,1001,1000), 'tfidf-weighted-w2v',closePlt=True)


# In[ ]:


for p in range(10, end_iteration, 10):
    genTSNEGif(review_vectors, len(review_vectors), p, range(1000,10001,1000), 'tfidf-weighted-w2v',closePlt=True)


# In[ ]:


#for p in range(10, end_iteration, 10):
    #genTSNEGif(review_vectors, len(review_vectors), p, range(1000,10001,1000), 'tfidf-weighted-w2v',closePlt=True)



# # TFIDF

# In[ ]:


# Create Vectors

tf_idf_vec = TfidfVectorizer(ngram_range=(1,2))
final_counts = tf_idf_vec.fit_transform(final_reviews['CleanedText'].values)

#.fit_transform(final_reviews['CleanedText'].values)
print('Shape of TF-IDF Vectorizer: ', final_counts.get_shape())
print('Total no.of unique words: ', final_counts.get_shape()[1])

# Standardize the Data
standardized_data = StandardScaler().fit_transform(final_counts.toarray().astype(np.float64)) #, with_mean=False
print('Shape of Standardized data', standardized_data.shape)


# In[ ]:


# On Standardized data
genTSNEGif(standardized_data, training_data_set_size, 30, range(1000,10001,1000), 'tfidf-std',closePlt=True)


# In[ ]:


# on Non-Standardization
dense_mat = final_counts.toarray().astype(np.float64)
for p in range(60, end_iteration, 10):
    genTSNEGif(dense_mat, training_data_set_size, p, range(1000,10001,1000), 'tfidf',closePlt=True)
