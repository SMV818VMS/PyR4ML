
# coding: utf-8

# # Random Forest Classifier in Python
# 
# In this notebook you will find the steps to solve a Random Forest classification problem in python.
# 
# ### Loading packages

# In[77]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
from numpy import genfromtxt, savetxt
import seaborn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import StratifiedKFold
from scipy import interp
import matplotlib.pyplot as plt


# ### Defining datasets
# 
# Define the test and training set we are going to use:

# In[78]:

#create the training & test sets, skipping the header row with [1:]
dataset = genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]    
target = [x[0] for x in dataset]
train = [x[1:] for x in dataset]
test = genfromtxt(open('Data/test.csv','r'), delimiter=',', dtype='f8')[1:]
test = [x[1:] for x in test]


# ### Generating a toy model dataset
# 
# We are going to generate here a dataset to run a toy model:

# In[79]:

#create and train the random forest
#multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf.fit(train, target)


# In[80]:

savetxt('Data/classes_predicted.csv', rf.predict(test), delimiter=',', fmt='%f')
savetxt('Data/probabil_predicted.csv', rf.predict_proba(test), delimiter=',', fmt='%f')


# In[ ]:




# In[ ]:



