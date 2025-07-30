import numpy as np 
import pandas as pd


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
 
# %%
import seaborn as sn
import matplotlib.pyplot as plt
import nltk

# %%
dataset = pd.read_csv(r'C:\Users\Gautam Bijarniya\OneDrive\Desktop\NTCC\netflix_titles.csv')


dataset.head()

# %%
from ydata_profiling import ProfileReport
ProfileReport(dataset, title='Netflix Reviews - Report', progress_bar=False)



# %%
dataset = dataset[['title','director','listed_in','description']]
dataset.head()

# %% [markdown]
# finding misiing values 

# %%
dataset.isna().sum()

# %%
dataset.director.fillna("", inplace = True)


# %%
dataset['movie_info'] = dataset['director'] + ' ' + dataset['listed_in']+ ' ' + dataset['description']

# %%
dataset.head()

# %%
dataset  = dataset[['title','movie_info']]

# %%
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# %%
from nltk.corpus import stopwords
import string
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)


# %%
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

# %%
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    final_text = []
    for i in text.split():
         if i.strip().lower() not in stop:
                word = lemmatizer.lemmatize(i.strip())
                final_text.append(word.lower())
                
    return  " ".join(final_text)      
                

# %%
dataset.movie_info = dataset.movie_info.apply(lemmatize_words)

# %%
dataset.head()

# %%
from sklearn.feature_extraction.text import CountVectorizer
tf=CountVectorizer()

# %%
X=tf.fit_transform(dataset['movie_info'])  

# %%
from sklearn.metrics.pairwise import cosine_similarity

# %%
cosine_sim = cosine_similarity(X)

# %%
liked_movie = 'Kota Factory'

# %%
index_l = dataset[dataset['title'] == liked_movie].index.values[0]
similar_movies = list(enumerate(cosine_sim[index_l]))
sort_movies = sorted(similar_movies , key = lambda X:X[1] , reverse = True)
sort_movies.pop(0)
sort_movies = sort_movies[:10]

# %%
sort_movies

# %%
for movies in sort_movies:
    print(dataset.title[movies[0]])