
# coding: utf-8

# In[172]:

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from utilities import ConfusionMatrix


# In[139]:

df = pd.read_csv('./clean/ml_dataset_train-1111.csv',index_col=0)


# In[140]:

validate_len = int(0.8*len(df))


# In[141]:

X_train,Y_train = df.values[:validate_len].T[0], df.values[:validate_len].T[1]


# In[142]:

X_test, Y_test = df.values[validate_len:].T[0], df.values[validate_len:].T[1]


# In[164]:

mb = MultinomialNB(alpha=1)
cv = CountVectorizer()
kb = SelectKBest(chi2, k=30000)


# In[165]:

X_train2 = kb.fit_transform(cv.fit_transform(X_train), list(Y_train))
X_train2 = cv.fit_transform(X_train)


# In[166]:

mb.fit(X_train2, list(Y_train))


# In[167]:

X_test2 = kb.transform(cv.transform(X_test))
X_test2 = cv.transform(X_test)


# In[168]:

Y_predicted = mb.predict(X_test2)


# In[169]:

cm = ConfusionMatrix(Y_test, Y_predicted)


# In[170]:

cm.average_accuracy()


# In[171]:

cm.confusion_matrix


# In[152]:

df = pd.read_csv('./clean/ml_dataset_test_in-1111.csv', index_col=0)


# In[153]:

X_to_predict = kb.transform(cv.transform(df.values.T[0]))


# In[154]:

Y_to_predict_predicted = mb.predict(X_to_predict)


# In[155]:

Y_to_predict_predicted


# In[156]:

df2 = pd.DataFrame(Y_to_predict_predicted)


# In[158]:

df2.to_csv('./predicted-mnb.csv')


# In[ ]:



