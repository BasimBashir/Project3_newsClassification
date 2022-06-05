#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import naive_bayes, metrics, model_selection, preprocessing

# In[2]:


data = pd.read_csv("uci-news-aggregator.csv")
data.head()

# In[3]:


import string

punct = string.punctuation

# In[4]:


# dataCleaning
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from spacy.lang.en.stop_words import STOP_WORDS

# In[5]:


nlp = spacy.load('en_core_web_sm')
stopwords = list(STOP_WORDS)


# In[6]:


def text_data_cleaning(sentence):
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        if token.lemma_ != '-PRON-':
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)

    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in punct:
            cleaned_tokens.append(token)
    return cleaned_tokens


# In[8]:


print(text_data_cleaning("  this is the best in the BASIM"))

# In[16]:


X = data['TITLE']

# In[17]:


y = data['CATEGORY']

# In[19]:


tfidf = TfidfVectorizer(tokenizer=text_data_cleaning)

# In[13]:


text = input('enter news text relevant to business,science,entertainment, health: \n')

# In[20]:


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

# In[21]:


clf = Pipeline([('tfidf', tfidf), ('clf', model)])  # It takes alot of time to run
clf.fit(X, y)

pred = clf.predict([text])

if pred == 'b':
    print("Business News")
elif pred == 't':
    print("Science and Technology")
elif pred == 'e':
    print("Entertainment")
elif pred == 'm':
    print("Health")

# In[26]:


import joblib

joblib.dump(clf, 'news_classifier.pkl')

# In[ ]:
