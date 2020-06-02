#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
import keras as k
import tensorflow as tf
import pandas as pd
import os
import io
import random
import logging
import re
import numpy as np
import math
from keras.layers import Input, Dense, Embedding, Conv1D, Conv2D, MaxPool2D,MaxPool1D, Reshape, Flatten, Dropout, Concatenate, concatenate,BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import re
from pathlib import Path
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from os import walk
from os import listdir
from os.path import isfile, join
from os import path
import docx2txt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import plotly.express as px
from keras import regularizers
import keras as k
import string


# In[1]:


def predict_class(test,model,all): 
    i = 0
    Xf_test = pre_process_4test(np.array([test]))
    pred = model.predict(Xf_test)
    classes = ['athletics', 'cricket', 'football', 'rugby', 'tennis'] #  classes = list(pd.get_dummies(y_train_childl1).columns)
    max_index_col = idx_maxcol(pred)
   
    if all == 'X':
        for classes in classes:
            print(classes,'    ',':',pred[0][i])
            i = i + 1
    return classes[max_index_col]
    


# In[ ]:





# In[2]:


def pre_process_4test(input):
    
    max_words_per_doc = 1679
#Vectorize your text to numeric idex sequences
    tokenizer        = Tokenizer()      # top_maxwords_atcorpuslevel ->E.g.considers only top 30K words in dataset
    tokenizer.fit_on_texts(input)                               # fit (probably just creates indices etc, basically initializes and develops Tokenizer object for this text corpus)
    indices_of_words = tokenizer.texts_to_sequences(input)      # maps(just replaces) words with w-indices in each sentence (if run w/o fit gives error word_indices not found)
    encoded_input = pad_sequences(indices_of_words, max_words_per_doc,padding ='post') 

    return encoded_input


# In[4]:


def idx_maxcol(a):
#     a = [[1,2,3,4,5]]
#     a = np.array(a)
#     a.shape
#     np.argmax(a, axis=1)[0]
  max_index_col = np.argmax(a, axis=1)[0]
  return max_index_col

