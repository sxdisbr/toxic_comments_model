#!/usr/bin/env python
# coding: utf-8

# <h1>Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Preparation" data-toc-modified-id="Preparation-1"><span class="toc-item-num">1</span> Preparation</a></span></li><li><span><a href="#Conclusions" data-toc-modified-id="Conclusions-2"><span class="toc-item-num">2</span> Conclusions</a></span></li>

# # In search of a toxic comments model

# A online store launches a new service. Now users can edit and supplement product descriptions, as in wiki communities. That is, clients offer their edits and comment on the changes of others. The store needs a tool that will search for toxic comments and send them for moderation.
# 
# We will training a model to classify comments into positive and negative. At our disposal is a data set with markup on the toxicity of edits.
# 
# We are going to build a model with the value of the quality metric *F1* at least 0.75.
# 
# **Instructions for the implementation of the project**
# 
# 1. Upload and prepare the data.
# 2. Train different models.
# 3. Draw conclusions.
# 
# It is not necessary to use *BERT* to complete the project, but you can try.
# 
# **Data description**
# 
# The data is in the file `toxic_comments.csv'. The *text* column in it contains the comment text, and *toxic* is the target attribute.

# ## Подготовка

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm.notebook import tqdm
tqdm.pandas()
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt    
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from sklearn.metrics import f1_score
import re
import warnings
warnings.filterwarnings("ignore")
import gc


# In[2]:


data = pd.read_csv(r'C:\Users\pinos\Downloads/toxic_comments.csv', index_col=0)


# In[3]:


def get_info():
    return data.sample(), data.info()


# In[4]:


get_info()


# We have three columns, the first of which seems to be the index, the second is comments, and the third is a rating according to a positive or negative parameter, i.e. 1 and zero.

# In[5]:


data.isna().mean()


# No missing data in this dataset.

# In[6]:


# A function that returns the number of words in a string
def count_words(string):
# Splitting the word
    words = string.split()
    
    # Returns the number of words
    return len(words)

# This create new column with the number of words in the comments
data['transcript'] = data['text'].progress_apply(count_words)

# Printing the average number of words in comments
print(data['transcript'].mean())


# The average volume of comments is 67 words.

# In[7]:


toxic=data['toxic'].value_counts()


# In[8]:


data['toxic'].value_counts(normalize=True).plot(
    kind='bar',figsize=(20, 10), facecolor='red')
plt.title('Balance of class distribution in data')
plt.xlabel('Comments')
plt.show()
plt.show()


# It is noteworthy that the values of the column are in a clear imbalance, it seems that negative reviews prevail over positive ones. This should be taken into account when choosing a model for training.

# Now we're going to clean up the text a bit with another function.

# In[16]:


lemmatizer = WordNetLemmatizer()
def lemmatize(sentence):
    text = re.sub(r'[^a-zA-Z]', ' ', sentence)
    word_list = word_tokenize(text)
    lemmatized_output = " ".join([lemmatizer.lemmatize(w) for w in word_list])
    return " ".join(lemmatized_output.split())

data['lemmatized'] = data['text'].progress_apply(lemmatize)

print(data['lemmatized'][0])


# In[14]:


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


# We check that the function performs its task.

# In[17]:


give_it_a_try = 'Humans cannot communicate; not even their brains can communicate; not even their conscious minds can communicate. Only communication can communicate.'
print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(give_it_a_try)])


# In[18]:


corpus = list(data['text'])


# In[19]:


get_ipython().run_cell_magic('time', '', "\ndef systemlem(text):\n    words = []\n    for i in nltk.word_tokenize(text):\n        lem = lemmatizer.lemmatize(i, get_wordnet_pos(i))\n        words.append(lem)\n    return ' '.join(words) \n\nlemma = []\n\nfor i in tqdm(range(len(corpus))):\n    \n    lemma.append(systemlem(lemmatize(corpus[i])))\n    \ndata['lemma']=data['text'].progress_apply(lemmatize)\n")


# In[20]:


data_toxic = data['lemma'][data['toxic']==1]
text_cloud = ' '.join(data_toxic)
cloud = WordCloud(collocations=False).generate(text_cloud)
plt.figure(figsize=(20,10))
plt.imshow(cloud)
plt.axis('off')
plt.show()   


# In[21]:


data_positive = data['lemma'][data['toxic']==0]
text_cloud = ' '.join(data_positive)
cloud = WordCloud(collocations=False).generate(text_cloud)
plt.figure(figsize=(20,10))
plt.imshow(cloud)
plt.axis('off')
plt.show()   


# We are collecting data as a preliminary stage of training.

# In[22]:


target=data['toxic']
features=data.drop(['toxic'], axis=1)


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size = 0.20, random_state=42)


# We convert the data and apply stop_words.

# In[24]:


corpus_train = X_train['text']
corpus_test = X_test['text']
stop_words = set(stopwords.words('english')) 


# Clearing the received data from garbage.

# In[25]:


del corpus_train, corpus_test
gc.collect()


# We set the search parameters and run a grid search to find the best parameters.

# In[26]:


pipe = Pipeline([
    ('vect', CountVectorizer( 
                             dtype=np.float32 
                             )),
    ('tfidf', TfidfTransformer()),
    ('model',MultinomialNB())]) 


# In[27]:


parameters_NB = {  
'vect__max_features': (1, 1000)  
}  


# In[28]:


grid_search_NB = GridSearchCV(
    pipe,  
    param_grid=parameters_NB, 
    scoring = 'f1',
    n_jobs = 1,
    cv = 5
)


# In[29]:


get_ipython().run_cell_magic('time', '', "corpus_train=X_train['text']\ngrid_search_NB.fit(corpus_train, y_train)    \nprint('Best parameters: ', grid_search_NB.best_params_ ) \nprint('Best score F1 - NB:', grid_search_NB.best_score_ )\n")


# In[30]:


pipe.get_params().keys()


# The results are not very good, they are lower than required.

# In[31]:


corpus_train=X_train['text']


# In[32]:


get_ipython().run_cell_magic('time', '', "params={'model__random_state': [1, 42, 999, 123456], \n        }\npipeline = Pipeline([\n    ('vect', CountVectorizer( \n                             dtype=np.float32, \n                             )),\n    ('tfidf', TfidfTransformer()),\n    ('model',LogisticRegression(random_state=1, class_weight='balance'))])\ngrid = GridSearchCV(pipeline, cv=5, param_grid=params,\n                    scoring='f1')\ngrid.fit(corpus_train, y_train)\nprint('Best parameters LogisticRegression: ', grid.best_params_ ) \nprint('Best score F1 - LogisticRegression:', grid.best_score_ )\n")


# The logistic regression model is also estimated below the required parameters.

# In[33]:


get_ipython().run_cell_magic('time', '', "params={  \n    'model__random_state': [1,10],\n    \n }\npipeline = Pipeline([\n    ('vect', CountVectorizer(\n                              \n                             dtype=np.float32, \n                             )),\n    ('tfidf', TfidfTransformer()),\n    ('model',LGBMClassifier(scale_pos_weight=3, num_iterations=100\n                           ))])\ngrid = GridSearchCV(pipeline, cv=2, param_grid=params,\n                    scoring='f1')\ngrid.fit(corpus_train, y_train)\nprint('Best parameters LGBMClassifier: ', grid.best_params_) \nprint('Best score F1 - LGBMClassifier:', grid.best_score_)\n")


# In[34]:


pipeline.get_params().keys()


# The result of the GSM Classifier is more than 76 percent.

# In[35]:


estimator = CatBoostClassifier(random_state=1, 
                               scale_pos_weight=3, 
                               n_estimators=100, 
                               learning_rate=0.1, 
                               thread_count=-1,
                               depth=3)


# In[36]:


get_ipython().run_cell_magic('time', '', "params={'model__random_state': [1,10],\n        }       \npipeline = Pipeline([\n    ('vect', CountVectorizer(\n                             dtype=np.float32, \n                             )),\n    ('tfidf', TfidfTransformer()),\n    ('model', estimator)])\ngrid = GridSearchCV(pipeline, cv=5, param_grid=params,\n                    scoring='f1')\ngrid.fit(corpus_train, y_train)\nprint('Best parameters CatBoostClassifier: ', grid.best_params_ ) \nprint('Best score F1 - CatBoostClassifier: ', grid.best_score_ )\n")


# The CatBoost model with 67 percent also does not meet the minimum level required for this task.

# In[37]:


pipeline.get_params().keys()


# Мы перешли к тестированию лучшей модели, которая оказалась классификатором ЛГБМ.

# In[38]:


corpus_test = X_test['text']


# In[40]:


get_ipython().run_cell_magic('time', '', "params={  \n    'tfidf__smooth_idf': [True, False],\n     \n }\npipeline = Pipeline([\n    ('vect', CountVectorizer( \n                             dtype=np.float32,  \n                             \n                             )),\n    ('tfidf', TfidfTransformer()),\n    ('model',LGBMClassifier(random_state=1,\n                            scale_pos_weight=3, \n                            num_threads=4,\n                            learning_rate=0.2,\n                            max_depth=3,\n                            n_estimators=1000,\n                            num_iterations=700,\n                            boosting_type='gbdt',\n                            num_leaves=40,\n                            silent=True,                                                   \n                           ))])\ngrid = GridSearchCV(pipeline, cv=5, param_grid=params,\n                    scoring='f1')\ngrid.fit(corpus_train, y_train)\npredictions=grid.predict(corpus_test)\nprint('Best parameters LGBMClassifier: ', grid.best_params_) \nprint('Score F1 at the test phase - LGBMClassifier:', grid.best_score_)\ntest_score = f1_score(y_test, predictions)\nprint('Final score F1 : ', test_score)\n")


# ## Conclusions

# Three classification models - NB, Logistic regression and Catboost models, even with optimized parameter search, failed to achieve the criteria required to get 75 percent of the points in F1.
# 
# The NB model scored 57 points for the account, the LogisticRegression model scored more points-72 percent, and the Catboost model-67 percent.
# 
# The LightBoost model, however, was 76 percent higher than the criteria we set at the beginning of the study. At the testing stage and after a long and thorough search for hyperparameters, we managed to reach almost 79 percent.
# 
# The latter, therefore, is the model we recommend for analyzing comments.
