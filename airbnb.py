#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import re
import math
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('train.csv')
updated_df = df.drop('price', axis=1)
print ('There are {} rows and {} columns in the dataset'.format(*updated_df.shape))
updated_df.head(3)


# In[3]:


df.price


# In[4]:


df.columns


# In[5]:


# drop all the unnecessary columns
feature_to_keep = ['id','description','access','property_type','room_type','accommodates','bathrooms',
                  'bedrooms','bed_type','guests_included','extra_people','minimum_nights',
                  'maximum_nights','cancellation_policy', 'neighbourhood_group_cleansed','host_is_superhost']

new_df = updated_df[feature_to_keep]

# remove the dollar sign before "cleaning_fee", "extra_people", "price" and change the datatype to numerical variables
feature_to_remove_dollar = ['extra_people']
new_df[feature_to_remove_dollar] = new_df[feature_to_remove_dollar].replace('\$','',regex = True)


# In[6]:


new_df.isna().sum()


# In[7]:


# drop 308 rows with missing descriptions
new_df.loc[:,'description'] = new_df['description'].dropna()

# fill NaN with median value for 'bathrooms', 'bedrooms','price'
new_df.loc[:,'bathrooms'] = new_df.loc[:,'bathrooms'].fillna(new_df.loc[:,'bathrooms'].median())
new_df.loc[:,'bedrooms'] = new_df.loc[:,'bedrooms'].fillna(new_df.loc[:,'bedrooms'].median())
#new_df.loc[:, 'reviews_per_month'] = new_df.loc[:, 'reviews_per_month'].fillna(new_df.loc[:, 'reviews_per_month'].median())


# In[8]:


def guests(x):
    if x == 1:
        return 'one'
    elif x == 2:
        return 'two'
    elif x == 3: 
        return 'three'
    elif x == 4:
        return 'four'
    else:
        return 'many'


# In[9]:


saved = new_df['guests_included'].apply(guests)
new_df['guests_included'] = saved
new_df


# In[10]:


new_df['property_type'].value_counts()


# In[11]:


# merge small catergories in property_type into one category "Other"
Other = ['Serviced apartment','Guest suite','Other','Boutique hotel','Bed and breakfast','Resort','Hotel','Guesthouse',
        'Hostel','Bungalow','Villa','Tiny house','Aparthotel','Boat', 'Tent', 'Cottage','Camper/RV','Casa particular (Cuba)',        
'Island',                        
'Timeshare',                     
'Chalet',                        
'Cabin',                         
'Houseboat',                     
'Train', 'Cave', 'Nature lodge', 'Earth house', 'Castle']
new_df['property_type'].loc[new_df['property_type'].isin(Other)] = "Other"


# In[12]:


new_df['property_type'].value_counts()


# In[13]:


new_df['room_type'].value_counts()


# In[14]:


new_df['bed_type'].value_counts()


# In[15]:


# merge small catergories in bed_type into one category "No Bed"
Other = ['Futon','Pull-out Sofa','Airbed','Couch']
new_df['bed_type'].loc[new_df['bed_type'].isin(Other)] = "No Bed"


# In[16]:


new_df['bed_type'].value_counts()


# In[17]:


new_df['cancellation_policy'].value_counts()


# In[18]:


# merge small catergories in cancellation_policy into one category "Other"
Other = ['super_strict_60','super_strict_30','strict','long_term']
new_df['cancellation_policy'].loc[new_df['cancellation_policy'].isin(Other)] = "Other"


# In[19]:


new_df


# In[20]:


import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

import json
import pyLDAvis.gensim
pyLDAvis.enable_notebook()

from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.ldamodel import LdaModel


# In[21]:


def preprocess_text(corpus):
    processed_corpus = []
    english_words = set(nltk.corpus.words.words())
    english_stopwords = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'[A-Za-z|!]+')
    for row in corpus:
        sentences = []
        word_tokens = tokenizer.tokenize(row)
        word_tokens_lower = [t.lower() for t in word_tokens]
        word_tokens_lower_english = [t for t in word_tokens_lower if t in english_words or not t.isalpha()]
        word_tokens_no_stops = [t for t in word_tokens_lower_english if not t in english_stopwords]
        word_tokens_no_stops_lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in word_tokens_no_stops]
        for word in word_tokens_no_stops_lemmatized:
            if len(word) >= 2:
                sentences.append(word)
        processed_corpus.append(sentences)
    return processed_corpus

def pipline(processed_corpus):
    dictionary = Dictionary(processed_corpus)
    doc_term_matrix = [dictionary.doc2bow(listing) for listing in processed_corpus]
    return dictionary, doc_term_matrix

def lda_topic_model(doc_term_matrix,dictionary,num_topics = 3, passes = 2):
    LDA = LdaModel
    ldamodel = LDA(doc_term_matrix,num_topics = num_topics, id2word = dictionary, passes = passes)
    return ldamodel

def topic_feature(ldamodel, doc_term_matrix, df, new_col, num_topics):
    docTopicProbMat = ldamodel[doc_term_matrix]
    docTopicProbDf = pd.DataFrame(index = df.index, columns = range(0,num_topics))
    for i,doc in enumerate(docTopicProbMat):
        for topic in doc:
            docTopicProbDf.iloc[i,topic[0]] = topic[1]
    docTopicProbDf = docTopicProbDf.fillna(0)
    docTopicProbDf[new_col] = docTopicProbDf.idxmax(axis=1)
    df_topics = docTopicProbDf[new_col]
    df_new = pd.concat([df,df_topics],axis = 1)
    return df_new


# In[22]:


import nltk
nltk.download('words')


# In[23]:


nltk.download('wordnet')


# In[24]:


corpus_description = new_df['description'].astype(str)

# use nlp package to process the text in description
processed_corpus_description = preprocess_text(corpus_description)

# generate the doc_term_matrix for lda model
#dictionary_description, doc_term_matrix_description = pipline(processed_corpus_description)

# lda model for topic modeling
#ldamodel_description = lda_topic_model(doc_term_matrix_description,dictionary_description)

# add the topic feature to the dataframe
#final_df = topic_feature(ldamodel_description,doc_term_matrix_description,new_df,new_col = 'description_topic', num_topics =3)


# In[25]:


# generate the doc_term_matrix for lda model
dictionary_description, doc_term_matrix_description = pipline(processed_corpus_description)


# In[26]:


# lda model for topic modeling
ldamodel_description = lda_topic_model(doc_term_matrix_description,dictionary_description)


# In[27]:


# add the topic feature to the dataframe
final_df = topic_feature(ldamodel_description,doc_term_matrix_description, new_df,new_col = 'description_topic', num_topics =4)


# In[28]:


review_median = df['reviews_per_month'].median()
final_df['reviews_per_month'] = df['reviews_per_month'].fillna(review_median)
final_df['access'] = df['access']


# In[ ]:





# In[29]:


def match(x):
    matches = ["full", "whole", "entire", "WHOLE", "ENTIRE"]
    if any(a in x for a in matches):
        return True
    else:
        return False


# In[30]:


saved = df['access'].fillna('None')
df['access'] = saved
str_saved = df['access'].apply(match)


# In[31]:


final_df['new_access'] = str_saved
final_df.head()


# In[32]:


def bath_match(x):
    if x < 3:
        return 'not many'
    else:
        return 'many'


# In[33]:


def bed_match(x):
    if x < 5:
        return 'not many'
    else:
        return 'many'


# In[34]:


def acc_match(x):
    if x < 8:
        return 'not many'
    else:
        return 'many'


# In[35]:


bath_saved = df['bathrooms'].apply(bath_match)
final_df['bath'] = bath_saved

bed_saved = df['bedrooms'].apply(bed_match)
final_df['bed'] = bed_saved

acc_saved = df['accommodates'].apply(acc_match)
final_df['acc'] = acc_saved


# In[36]:


s_corpus_description = df['space'].astype(str)

# use nlp package to process the text in description
s_processed_corpus_description = preprocess_text(s_corpus_description)


# In[37]:


# generate the doc_term_matrix for lda model
s_dictionary_description, s_doc_term_matrix_description = pipline(s_processed_corpus_description)


# In[38]:


# lda model for topic modeling
s_ldamodel_description = lda_topic_model(s_doc_term_matrix_description, s_dictionary_description)


# In[39]:


# add the topic feature to the dataframe
the_final_df = topic_feature(s_ldamodel_description,s_doc_term_matrix_description, final_df,new_col = 'space_topic', num_topics =3)


# In[ ]:





# In[40]:


n_corpus_description = df['name'].astype(str)

# use nlp package to process the text in description
n_processed_corpus_description = preprocess_text(n_corpus_description)


# In[41]:


# generate the doc_term_matrix for lda model
n_dictionary_description, n_doc_term_matrix_description = pipline(n_processed_corpus_description)


# In[42]:


# lda model for topic modeling
n_ldamodel_description = lda_topic_model(n_doc_term_matrix_description, n_dictionary_description)


# In[43]:


# add the topic feature to the dataframe
name_final_df = topic_feature(n_ldamodel_description,n_doc_term_matrix_description, the_final_df,new_col = 'name_topic', num_topics =3)


# In[44]:


sum_corpus_description = df['summary'].astype(str)

# use nlp package to process the text in description
sum_processed_corpus_description = preprocess_text(sum_corpus_description)


# In[45]:


# generate the doc_term_matrix for lda model
sum_dictionary_description, sum_doc_term_matrix_description = pipline(sum_processed_corpus_description)


# In[46]:


# lda model for topic modeling
sum_ldamodel_description = lda_topic_model(sum_doc_term_matrix_description, sum_dictionary_description)


# In[47]:


# add the topic feature to the dataframe
summary_final_df = topic_feature(sum_ldamodel_description,sum_doc_term_matrix_description, name_final_df,new_col = 'summary_topic', num_topics =3)


# In[48]:


def accommodates(x):
    if x >= 7:
        return True
    else:
        return False


# In[49]:


accom_saved = df['accommodates'].apply(accommodates)
summary_final_df['acc_lot?'] = accom_saved
summary_final_df.head(3)


# In[503]:


data = df.groupby(['host_neighbourhood']).mean()['price'].sort_values()

really_cheap = data.loc[(data.values >= 0) & (data.values < 50)].index.tolist()
pretty_cheap = data.loc[(data.values >= 50) & (data.values < 75)].index.tolist()
cheap = data.loc[(data.values >= 75) & (data.values < 100)].index.tolist()
under_110 = data.loc[(data.values >= 100) & (data.values < 110)].index.tolist()
under_120 = data.loc[(data.values >= 110) & (data.values < 120)].index.tolist()
under_130 = data.loc[(data.values >= 120) & (data.values < 130)].index.tolist()
under_140 = data.loc[(data.values >= 130) & (data.values < 140)].index.tolist()
under_150 = data.loc[(data.values >= 140) & (data.values < 150)].index.tolist()
under_160 = data.loc[(data.values >= 150) & (data.values < 160)].index.tolist()
under_180 = data.loc[(data.values >= 160) & (data.values < 180)].index.tolist()
under_190 = data.loc[(data.values >= 180) & (data.values < 190)].index.tolist()
under_200 = data.loc[(data.values >= 190) & (data.values < 200)].index.tolist()
under_210 = data.loc[(data.values >= 200) & (data.values < 210)].index.tolist()
under_220 = data.loc[(data.values >= 210) & (data.values < 220)].index.tolist()
under_230 = data.loc[(data.values >= 220) & (data.values < 230)].index.tolist()
under_240 = data.loc[(data.values >= 230) & (data.values < 240)].index.tolist()
under_250 = data.loc[(data.values >= 240) & (data.values < 250)].index.tolist()
under_260 = data.loc[(data.values >= 250) & (data.values < 260)].index.tolist()
under_270 = data.loc[(data.values >= 260) & (data.values < 270)].index.tolist()
under_280 = data.loc[(data.values >= 270) & (data.values < 280)].index.tolist()
expensive_300 = data.loc[(data.values >= 280) & (data.values < 400)].index.tolist()
expensive_400 = data.loc[(data.values >= 400) & (data.values < 500)].index.tolist()
expensive_500 = data.loc[(data.values >= 500) & (data.values < 600)].index.tolist()
expensive_600 = data.loc[(data.values >= 600) & (data.values < 700)].index.tolist()
really_expensive = data.loc[data.values >= 700].index.tolist()


# In[465]:


def host_cat(x):
    if x in really_cheap:
        return 'really cheap'
    elif x in pretty_cheap:
        return 'pretty cheap'
    elif x in cheap:
        return 'cheap'
    elif x in under_110:
        return 'under 110'
    elif x in under_120:
        return 'under 120'
    elif x in under_130:
        return 'under 130'
    elif x in under_140:
        return 'under 140'
    elif x in under_150:
        return 'under 150'
    elif x in under_160:
        return 'under 160'
    elif x in under_180:
        return 'under 180'
    elif x in under_190:
        return 'under 190'
    elif x in under_200:
        return 'under 200'
    elif x in under_110:
        return 'under 210'
    elif x in under_220:
        return 'under 220'
    elif x in under_230:
        return 'under 230'
    elif x in under_240:
        return 'under 240'
    elif x in under_250:
        return 'under 250'
    elif x in under_260:
        return 'under 260'
    elif x in under_270:
        return 'under 270'
    elif x in under_280:
        return 'under 280'
    elif x in expensive_300:
        return 'expensive_300'
    elif x in expensive_400:
        return 'expensive_400'
    elif x in expensive_500:
        return 'expensive_500'
    elif x in expensive_600:
        return 'expensive_600'
    else:
        return 'really expensive'


# In[466]:


host_saved = df['host_neighbourhood'].apply(host_cat)
summary_final_df['host_cat'] = host_saved
#summary_final_df.head()


# In[467]:


data = df.groupby(['neighbourhood_cleansed']).mean().price.sort_values()


# In[468]:


under_50 = data.loc[(data.values >= 0) & (data.values < 50)].index.tolist()
under_100 = data.loc[(n_data.values >= 50) & (data.values < 100)].index.tolist()
under_125 = data.loc[(data.values >= 100) & (data.values < 125)].index.tolist()
under_150 = data.loc[(data.values >= 125) & (data.values < 150)].index.tolist()
under_175 = data.loc[(data.values >= 150) & (data.values < 175)].index.tolist()
under_200 = data.loc[(data.values >= 175) & (data.values < 200)].index.tolist()
under_225 = data.loc[(data.values >= 200) & (data.values < 225)].index.tolist()
under_250 = data.loc[(data.values >= 225) & (data.values < 250)].index.tolist()
under_275 = data.loc[(data.values >= 250) & (data.values < 275)].index.tolist()
under_300 = data.loc[(data.values >= 275) & (data.values < 300)].index.tolist()
under_400 = data.loc[(data.values >= 300) & (data.values < 400)].index.tolist()
over_400 = data.loc[(data.values >= 400) & (data.values < 800)].index.tolist()


# In[469]:


def n_cat(x):
    if x in under_50:
        return 'under 50'
    elif x in under_100:
        return 'under 100'
    elif x in under_125:
        return 'under 125'
    elif x in under_150:
        return 'under 150'
    elif x in under_175:
        return 'under 175'
    elif x in under_200:
        return 'under 200'
    elif x in under_225:
        return 'under 225'
    elif x in under_250:
        return 'under 250'
    elif x in under_275:
        return 'under 275'
    elif x in under_300:
        return 'under 300'
    elif x in under_400:
        return 'under 400'
    else:
        return 'over 400'


# In[470]:


n_saved = df['neighbourhood_cleansed'].apply(n_cat)
summary_final_df['n_cat'] = n_saved


# In[157]:


df.groupby(['guests_included']).mean().price


# In[158]:


def guest_classify(x):
    if (x >= 1) and (x < 4):
        return 'pretty cheap'
    elif (x >= 4) and (x < 8):
        return 'medium'
    else:
        return 'expensive'


# In[165]:


guest_saved = df['guests_included'].apply(guest_classify)
summary_final_df['guest_cat'] = guest_saved
guest_saved.value_counts()


# In[172]:


df.groupby(['number_of_reviews']).mean().price.head(5)


# In[173]:


def reviews_num(x):
    if x < 170:
        return 'three_digit'
    else:
        return 'two_digit'


# In[174]:


review_saved = df['number_of_reviews'].apply(reviews_num)
summary_final_df['review_cat'] = review_saved


# In[200]:


df.groupby(['reviews_per_month']).mean().price.head(670)


# In[201]:


def rpm_cat(x):
    if (x >= 0.00) & (x < 4.97):
        return 'high'
    elif (x >= 4.97) & (x < 6.5):
        return 'medium'
    else:
        return 'low'


# In[203]:


rpm_saved = summary_final_df['reviews_per_month'].apply(reviews_num)
summary_final_df['rpm_cat'] = rpm_saved


# In[214]:


combined = df['accommodates'] + df['bedrooms'] + df['bathrooms']
com_df = pd.DataFrame(combined)
com_df['price'] = df['price']
com_df.groupby([0]).mean()


# In[505]:


def combined_cat(x):
    if x <= 3:
        return 'one'
    elif (x > 3) & (x <= 5):
        return 'two'
    elif (x > 5) & (x <= 7):
        return 'three'
    elif (x > 7) & (x <= 10):
        return 'four'
    elif (x > 10) & (x <= 12):
        return 'four'
    elif (x > 12) & (x <= 15):
        return 'four'
    else:
        return 'else'


# In[506]:


combined_saved = com_df[0].apply(combined_cat)
summary_final_df['combined_cat'] = combined_saved


# In[ ]:





# In[ ]:





# In[513]:


cols_to_keep = [
'accommodates','bathrooms','bedrooms','maximum_nights','minimum_nights','property_type', 'reviews_per_month',
                'bed_type','room_type','extra_people','new_access','cancellation_policy', 'acc_lot?', 'host_cat', 'n_cat',
                'neighbourhood_group_cleansed', 'description_topic', 'space_topic','name_topic', 'summary_topic', 
    'host_is_superhost', 'guest_cat', 'review_cat']
model_df = summary_final_df[cols_to_keep]

# convert strings to dummies
categorical_feats = ['property_type','room_type','bed_type', 'new_access','cancellation_policy',
                     'neighbourhood_group_cleansed',
                     'description_topic', 'space_topic','name_topic', 'summary_topic', 'acc_lot?', 
                     'host_cat', 'n_cat', 'host_is_superhost', 'guest_cat', 'review_cat']
model_df = pd.get_dummies(model_df,columns = categorical_feats,drop_first = False)

# separate the target variable "yield" from the dataset
target = df['price']
X_df = model_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[96]:


test_df = pd.read_csv('test.csv')
print ('There are {} rows and {} columns in the dataset'.format(*test_df.shape))
test_df.head(3)


# In[97]:


# drop all the unnecessary columns
test_feature_to_keep = ['id','description','property_type','room_type','accommodates','bathrooms',
                  'bedrooms','bed_type','guests_included','extra_people','minimum_nights',
                  'maximum_nights','reviews_per_month', 'cancellation_policy','neighbourhood_group_cleansed','host_is_superhost']

test_new_df = test_df[test_feature_to_keep]

# remove the dollar sign before "cleaning_fee", "extra_people", "price" and change the datatype to numerical variables
test_feature_to_remove_dollar = ['extra_people']
test_new_df.loc[:,test_feature_to_remove_dollar] = test_new_df.loc[:,test_feature_to_remove_dollar].replace('\$','',regex = True)
test_new_df.loc[:,test_feature_to_remove_dollar] = test_new_df.loc[:,test_feature_to_remove_dollar].apply(pd.to_numeric,errors = "coerce")


# In[98]:


test_new_df.isna().sum()


# In[99]:


# drop 159 rows with missing descriptions
test_new_df.loc[:,'description'] = test_new_df['description'].dropna()

# fill NaN with median value for 'bathrooms', 'bedrooms','price'
test_new_df.loc[:,'bathrooms'] = test_new_df.loc[:,'bathrooms'].fillna(test_new_df.loc[:,'bathrooms'].median())
test_new_df.loc[:,'bedrooms'] = test_new_df.loc[:,'bedrooms'].fillna(test_new_df.loc[:,'bedrooms'].median())
test_new_df.loc[:,'reviews_per_month'] = test_new_df.loc[:,'reviews_per_month'].fillna(test_new_df.loc[:,'reviews_per_month'].median())


# In[100]:


test_new_df['property_type'].value_counts()


# In[101]:


# merge small catergories in property_type into one category "Other"
test_Other = ['Serviced apartment','Bed and breakfast','Resort','Hotel','Guesthouse','Other','Hostel', 'Guest suite', 'Boutique hotel', 
'Bungalow','Villa','Tiny house','Aparthotel','Boat', 'Tent', 'Cottage','Camper/RV','Casa particular (Cuba)', 'Nature lodge',       
'Island','Castle', 'Earth house','Cabin', 'Cave']

test_new_df['property_type'].loc[test_new_df['property_type'].isin(test_Other)] = "Other"


# In[102]:


test_new_df['property_type'].value_counts()


# In[103]:


test_new_df['room_type'].value_counts()


# In[104]:


test_new_df['bed_type'].value_counts()


# In[105]:


# merge small catergories in bed_type into one category "No Bed"
Other = ['Futon','Pull-out Sofa','Airbed','Couch']
test_new_df['bed_type'].loc[test_new_df['bed_type'].isin(Other)] = "No Bed"


# In[106]:


test_new_df['bed_type'].value_counts()


# In[107]:


test_new_df['cancellation_policy'].value_counts()


# In[108]:


# merge small catergories in cancellation_policy into one category "Other"
Other = ['super_strict_60','super_strict_30','strict','long_term']
test_new_df['cancellation_policy'].loc[test_new_df['cancellation_policy'].isin(Other)] = "Other"


# In[109]:


test_new_df['cancellation_policy'].value_counts()


# In[110]:


test_corpus_description = test_new_df['description'].astype(str)

# use nlp package to process the text in description
test_processed_corpus_description = preprocess_text(test_corpus_description)


# In[111]:


# generate the doc_term_matrix for lda model
test_dictionary_description, test_doc_term_matrix_description = pipline(test_processed_corpus_description)


# In[112]:


# lda model for topic modeling
test_ldamodel_description = lda_topic_model(test_doc_term_matrix_description, test_dictionary_description)


# In[113]:


# add the topic feature to the dataframe
test_final_df = topic_feature(test_ldamodel_description, test_doc_term_matrix_description, test_new_df, new_col = 'description_topic', num_topics =4)


# In[114]:


review_median = test_df['reviews_per_month'].median()
test_final_df['reviews_per_month'] = test_df['reviews_per_month'].fillna(review_median)
test_final_df['access'] = test_df['access']


# In[ ]:





# In[115]:


test_saved =test_df['access'].fillna('None')
test_df['access'] = saved
test_str_saved = test_df['access'].apply(match)


# In[116]:


test_final_df['new_access'] = test_str_saved


# In[117]:


test_bath_saved = test_df['bathrooms'].apply(bath_match)
test_final_df['bath'] = test_bath_saved

test_bed_saved = test_df['bedrooms'].apply(bed_match)
test_final_df['bed'] = test_bed_saved

test_acc_saved = test_df['accommodates'].apply(acc_match)
test_final_df['acc'] = test_acc_saved


# In[118]:


test_s_corpus_description = test_df['space'].astype(str)

# use nlp package to process the text in description
test_s_processed_corpus_description = preprocess_text(test_s_corpus_description)


# In[119]:


# generate the doc_term_matrix for lda model
test_s_dictionary_description, test_s_doc_term_matrix_description = pipline(test_s_processed_corpus_description)


# In[120]:


# lda model for topic modeling
test_s_ldamodel_description = lda_topic_model(test_s_doc_term_matrix_description, test_s_dictionary_description)


# In[121]:


# add the topic feature to the dataframe
test_the_final_df = topic_feature(test_s_ldamodel_description,test_s_doc_term_matrix_description, test_final_df,new_col = 'space_topic', num_topics =3)


# In[128]:


test_n_corpus_description = test_df['name'].astype(str)
# use nlp package to process the text in description
test_n_processed_corpus_description = preprocess_text(test_n_corpus_description)


# In[129]:


# generate the doc_term_matrix for lda model
test_n_dictionary_description, test_n_doc_term_matrix_description = pipline(test_n_processed_corpus_description)


# In[130]:


# lda model for topic modeling
test_n_ldamodel_description = lda_topic_model(test_n_doc_term_matrix_description, test_n_dictionary_description)


# In[131]:


# add the topic feature to the dataframe
test_name_final_df = topic_feature(test_n_ldamodel_description,test_n_doc_term_matrix_description, test_the_final_df,new_col = 'name_topic', num_topics =3)


# In[132]:


test_sum_corpus_description = test_df['summary'].astype(str)
# use nlp package to process the text in description
test_sum_processed_corpus_description = preprocess_text(test_sum_corpus_description)


# In[133]:


# generate the doc_term_matrix for lda model
test_sum_dictionary_description, test_sum_doc_term_matrix_description = pipline(test_sum_processed_corpus_description)


# In[134]:


# lda model for topic modeling
test_sum_ldamodel_description = lda_topic_model(test_sum_doc_term_matrix_description, test_sum_dictionary_description)


# In[135]:


# add the topic feature to the dataframe
test_summary_final_df = topic_feature(test_sum_ldamodel_description,test_sum_doc_term_matrix_description, test_name_final_df,new_col = 'summary_topic', num_topics =3)


# In[136]:


test_accom_saved = test_df['accommodates'].apply(accommodates)
test_summary_final_df['acc_lot?'] = test_accom_saved
test_summary_final_df.head(3)


# In[474]:


test_host_saved = test_df['host_neighbourhood'].apply(host_cat)
test_summary_final_df['host_cat'] = test_host_saved
#summary_final_df.head()


# In[475]:


test_n_saved = test_df['neighbourhood_cleansed'].apply(n_cat)
test_summary_final_df['n_cat'] = test_n_saved


# In[457]:


test_guest_saved = test_df['guests_included'].apply(guest_classify)
test_summary_final_df['guest_cat'] = test_guest_saved


# In[458]:


test_review_saved = test_df['number_of_reviews'].apply(reviews_num)
test_summary_final_df['review_cat'] = test_review_saved


# In[ ]:





# In[ ]:





# In[476]:


test_cols_to_keep = ['accommodates','bathrooms','bedrooms','extra_people','maximum_nights','minimum_nights','property_type',
                'bed_type','room_type','cancellation_policy', 'reviews_per_month', 'new_access', 'acc_lot?', 'host_cat', 'n_cat',
                     'neighbourhood_group_cleansed', 'description_topic', 'space_topic', 'name_topic', 'summary_topic', 
                     'host_is_superhost', 'guest_cat', 'review_cat']
test_model_df = test_summary_final_df[test_cols_to_keep]

# convert strings to dummies
test_categorical_feats = ['property_type','room_type','bed_type', 'new_access','cancellation_policy', 'acc_lot?',
                           'description_topic', 'space_topic', 'name_topic', 'summary_topic', 'host_cat', 'n_cat',
                          'host_is_superhost', 'neighbourhood_group_cleansed', 'guest_cat', 'review_cat']
test_model_df = pd.get_dummies(test_model_df,columns = test_categorical_feats,drop_first = False)

test_X_df = test_model_df


# In[477]:


print(test_X_df.shape)
print(X_df.shape)


# In[479]:


test_X_df = test_X_df[['accommodates', 'bathrooms', 'bedrooms', 'maximum_nights',
       'minimum_nights', 'reviews_per_month', 'extra_people',
       'property_type_Apartment', 'property_type_Condominium',
       'property_type_House', 'property_type_Loft', 'property_type_Other',
       'property_type_Townhouse', 'room_type_Entire home/apt',
       'room_type_Private room', 'room_type_Shared room', 'bed_type_No Bed',
       'bed_type_Real Bed', 'new_access_False', 'new_access_True',
       'cancellation_policy_Other', 'cancellation_policy_flexible',
       'cancellation_policy_moderate',
       'cancellation_policy_strict_14_with_grace_period',
       'neighbourhood_group_cleansed_Bronx',
       'neighbourhood_group_cleansed_Brooklyn',
       'neighbourhood_group_cleansed_Manhattan',
       'neighbourhood_group_cleansed_Queens',
       'neighbourhood_group_cleansed_Staten Island', 'description_topic_0',
       'description_topic_1', 'description_topic_2', 'space_topic_0',
       'space_topic_1', 'space_topic_2', 'name_topic_0', 'name_topic_1',
       'name_topic_2', 'summary_topic_0', 'summary_topic_1', 'summary_topic_2',
       'acc_lot?_False', 'acc_lot?_True', 'host_cat_cheap',
       'host_cat_expensive_300', 'host_cat_expensive_400',
       'host_cat_expensive_500', 'host_cat_pretty cheap',
       'host_cat_really cheap', 'host_cat_really expensive',
       'host_cat_under 110', 'host_cat_under 120', 'host_cat_under 130',
       'host_cat_under 140', 'host_cat_under 150', 'host_cat_under 160',
       'host_cat_under 180', 'host_cat_under 190', 'host_cat_under 200',
       'host_cat_under 220', 'host_cat_under 230', 'host_cat_under 240',
       'host_cat_under 250', 'host_cat_under 260', 'host_cat_under 270',
       'host_cat_under 280', 'n_cat_over 400', 'n_cat_under 100',
       'n_cat_under 125', 'n_cat_under 150', 'n_cat_under 175',
       'n_cat_under 200', 'n_cat_under 225', 'n_cat_under 250',
       'n_cat_under 275', 'n_cat_under 300', 'n_cat_under 400',
       'n_cat_under 50', 'host_is_superhost_f', 'host_is_superhost_t',
       'guest_cat_expensive', 'guest_cat_medium', 'guest_cat_pretty cheap',
       'review_cat_three_digit', 'review_cat_two_digit']]


# In[508]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

#seed = 42
#X_train,X_test,y_train,y_test = train_test_split(X_df,target,random_state=seed)
linreg = LinearRegression(n_jobs=5).fit(X_df, target)
y_pred_linreg = linreg.predict(X_df)
print("Root Mean squared error: %.3f" %np.sqrt(mean_squared_error(target,y_pred_linreg)))
#rmse_lr = np.sqrt(mean_squared_error(target,y_pred_linreg))
#print(X_df.columns)
#print(test_X_df.columns)
y_pred_linreg


# In[514]:


from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=100).fit(X_df, target)
y_pred_rf = rf_reg.predict(X_df)
print("Root Mean squared error: %.3f" %np.sqrt(mean_squared_error(target,y_pred_rf)))
#rmse_rf = np.sqrt(mean_squared_error(y_test,y_pred_linreg))
y_pred_rf


# In[530]:


from sklearn.ensemble import GradientBoostingRegressor

de_reg = GradientBoostingRegressor(n_estimators=220).fit(X_df, target)
y_pred_dt = de_reg.predict(test_X_df)
#print("Root Mean squared error: %.3f" %np.sqrt(mean_squared_error(target,y_pred_dt)))
y_pred_dt


# In[502]:


from sklearn.tree import DecisionTreeRegressor

t_reg = DecisionTreeRegressor().fit(X_df, target)
y_pred_t = t_reg.predict(X_df)
print("Root Mean squared error: %.3f" %np.sqrt(mean_squared_error(target,y_pred_t)))
y_pred_t


# In[531]:


output_test = pd.DataFrame()
output_test['Id'] = test_df.id
output_test['Predicted'] = y_pred_dt
output_test.to_csv('test_pred_38.csv', index = False)


# In[ ]:





# In[ ]:




