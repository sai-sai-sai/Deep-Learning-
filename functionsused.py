#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


# fnam = r'C:\D Drive\My Study\Sturctured Content\NLP\Data\Class Docs\bbcsport-fulltext\bbcsport\athletics\001.txt'
# mode = 'r'
# file_object = open(fnam,mode)
# body = file_object.read()    
# body


# In[ ]:


def read_files(fnam,mode):
    type = os.path.splitext(fnam)
    if type[1] == '.txt':
        file_object = open(fnam,mode)
        body = file_object.read()
#         print(body)
#     elif type[1] == '.pdf':
#         a = 1
    elif type[1] == '.docx':
        body = docx2txt.process(fnam)
    else:
        print("Check supported filetype %s", fanm)
    return body


# In[ ]:


def walk_for_traingdata(top_folder_path):
    X = pd.DataFrame()
    f = []
    i = 0
    for (dirpath, dirnames, filenames) in walk(top_folder_path):
        i += 1
        #print( i, "Dirpath:" , dirpath, "Dirnames:" dirnames, "fnames:" filenames)
#         print(i, 'Dirpath is %s.' % dirpath , 'Dirname is %s.' % dirnames , 'fnames is %s.' % filenames ) 
        if dirnames ==  []:        
            for f in filenames:
                word_list = dirpath.split("\\")
#                 print(word_list[-1] )
                fullfname = join(dirpath,f)
                body = read_files(fullfname,'r')
#                 print("body IS ", body)
                dict_temp = { "Body" : body , "Parent Class L1" : word_list[-2], "Child Class L1" : word_list[-1] } 
#                 print("DT IS ", dict_temp)
                X = X.append(dict_temp,ignore_index=True) 
    X['Body'] = X['Body'].apply(remove_punctuations)
    X.apply(np.random.shuffle, axis=0)
    X= X.applymap(lambda s:s.lower() if type(s) == str else s)
    print(X[0:20])
    X_train = X["Body"] 
    y_train_parentl1 = X['Parent Class L1']
    y_train_childl1  = X['Child Class L1']
    print("Shape of X_train" , X_train.shape)
    print("Shape of Y_train parent and child" , X_train.shape)
    return X,X_train,y_train_parentl1,y_train_childl1


# In[ ]:


def analyze_words_tokenizer_vs_originalset(word_index_dict,unique_words) :
    words_from_tokenizer_notin_oringinal_set= []
    words_from_oringinal_set_notin_tokenizer = []
    words_4m_tokenizer  = list(word_index_dict.keys())
    list_unique_words = list(unique_words[0])
    for i in words_4m_tokenizer:
        if i not in list_unique_words:
            words_from_tokenizer_notin_oringinal_set.append(i)
    for i in list_unique_words:
        if i not in words_4m_tokenizer:
            words_from_oringinal_set_notin_tokenizer.append(i)
    
    words_from_tokenizer_notin_oringinal_set = pd.DataFrame(words_from_tokenizer_notin_oringinal_set)
    words_from_oringinal_set_notin_tokenizer = pd.DataFrame(words_from_oringinal_set_notin_tokenizer)
    words_from_tokenizer_notin_oringinal_set.to_excel(r"C:\D Drive\My Study\Sturctured Content\NLP\Data\Class Docs\bbcsport-fulltext\bbcsport\words_from_tokenizer_notin_oringinal_set.xlsx")
    words_from_oringinal_set_notin_tokenizer.to_excel(r"C:\D Drive\My Study\Sturctured Content\NLP\Data\Class Docs\bbcsport-fulltext\bbcsport\words_from_oringinal_set_notin_tokenizer.xlsx")

    
    return words_from_tokenizer_notin_oringinal_set,words_from_oringinal_set_notin_tokenizer


# In[ ]:


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


# In[ ]:


# mypath = r'C:\D Drive\My Study\Sturctured Content\NLP\Data\Class Docs\bbcsport-fulltext\bbcsport'
# X_train,y_train_parentl1,y_train_childl1 = walk_for_traingdata(mypath)


# In[ ]:


def pre_process(input,y_train_childl1,max_words_per_doc,top_maxwords_atcorpuslevel):
    
    top_maxwords_atcorpuslevel      =  top_maxwords_atcorpuslevel # E.g.consider max vocab of 30K words in whole corpus
    max_words_per_doc               =  int(max_words_per_doc)     # consider only 100 words per doc/review
#Vectorize your text to numeric idex sequences
    tokenizer        = Tokenizer()      # top_maxwords_atcorpuslevel ->E.g.considers only top 30K words in dataset
    tokenizer.fit_on_texts(input)                               # fit (probably just creates indices etc, basically initializes and develops Tokenizer object for this text corpus)
    indices_of_words = tokenizer.texts_to_sequences(input)      # maps(just replaces) words with w-indices in each sentence (if run w/o fit gives error word_indices not found)
    encoded_input = pad_sequences(indices_of_words, max_words_per_doc,padding ='post') 
#What words assigned to what indexes in entire corpus
    word_index_dict  = tokenizer.word_index                       # dictionary with word and its index 
#Target    
    yf_train = pd.get_dummies(y_train_childl1)
    yf_train = np.array(yf_train)
    return encoded_input,yf_train, word_index_dict, indices_of_words


# In[ ]:


def pre_process_4test(input):
    
    max_words_per_doc = 2000
#Vectorize your text to numeric idex sequences
    tokenizer        = Tokenizer()      # top_maxwords_atcorpuslevel ->E.g.considers only top 30K words in dataset
    tokenizer.fit_on_texts(input)                               # fit (probably just creates indices etc, basically initializes and develops Tokenizer object for this text corpus)
    indices_of_words = tokenizer.texts_to_sequences(input)      # maps(just replaces) words with w-indices in each sentence (if run w/o fit gives error word_indices not found)
    encoded_input = pad_sequences(indices_of_words, max_words_per_doc,padding ='post') 

    return encoded_input


# In[ ]:


def remove_by_patterns(X_train):
    X_train.str.replace(pat = '\n\n*', repl=' ',regex=True)
    regex_pat = re.compile(r'\n\n.', flags=re.IGNORECASE)
    X_train.str.replace(regex_pat, ' ')


# In[ ]:


def get_vocabsize(X_train):
    s = ''
    for i in X_train:
        s = i + s
    unique_words = [set(s.split())]
    num_unique_words = len(set(s.split()))
    return unique_words,num_unique_words


# In[ ]:


def corpus_details(X_train,y_train_parentl1,y_train_childl1):
  
    total_numberof_docs          = X_train.shape[0]    
    lengthsofdocs                = X_train.map(lambda x: len(x.split()))#number of words per doc/row
    total_num_ofwords            = lengthsofdocs.sum()
    unique_words,uniquewords_actual_vocabsize = get_vocabsize(X_train)
    rowindex_withmaxwords        = lengthsofdocs.idxmax()
    stats                        = lengthsofdocs.describe()
    max_words_per_doc            = int(stats[7])#stats[7] same as X_train.map(lambda x: len(x.split())).max()
    num_classes_childl1          = y_train_childl1.unique().shape[0] #len(y_train_childl1["Child Class L1"].unique()
    num_classes_parentl1         = y_train_parentl1.unique().shape[0] #len(y_train_childl1["Parent Class L1"].unique()
    perc = (uniquewords_actual_vocabsize*100/total_num_ofwords)
    
    print("************About the Corpus*****************")
    print("Total Number of Docs  in the corpus are                          " ,  total_numberof_docs)
    print("Total Number of words in the corpus are                          " ,  total_num_ofwords)
    print("Total Number of unique words in the corpus are(Vocab Size )      " , uniquewords_actual_vocabsize)
    print ('Percent of vocab size to total num of words is just               %.f %% ' % perc) 
    print("Longest(by words) document index                                 " ,  rowindex_withmaxwords)
    print("Max Number of words in a doc in across corpus(in the longest doc)" ,  stats[7])
    print("Total Number of Unique L1Child Classes in corpus                 " , num_classes_childl1)
    print("Total Number of Unique L1Parent Classes in corpuss               " , num_classes_parentl1)
    return total_num_ofwords,uniquewords_actual_vocabsize,max_words_per_doc,num_classes_childl1


# In[ ]:


def get_glove_embeddings(glove_dir):
    actual_glove_file  = os.path.join(glove_dir, 'glove.6B.100d.txt') # just glove file
#    actual_glove_file_path = open(os.path.join(glove_dir, 'glove.6B.100d.txt'),encoding="utf8") # open the glove file for readinf
#Construct Embeddings_Index from the Embedding file on disk : This will take some time run as embedding file has 400K words
    embeddings_index_dict = {}
    with open(actual_glove_file, mode = 'rb') as f:
        for i, line in enumerate(f):
                values = line.split()
                word = values[0].decode("utf-8")
                coefs = np.asarray(values[1:],dtype='float32')
                embeddings_index_dict[word] = coefs
    return embeddings_index_dict


# In[ ]:


def mapwordembeddings(word_index_dict,embeddings_index_dict,top_maxwords_atcorpuslevel,embedding_dim):
    #Create Embeding matrix with zeroes 
    embedding_matrix = np.zeros((top_maxwords_atcorpuslevel,embedding_dim))   # notice np.zeroes(inner tuple bracket), #type array
                                                                 # notice also the embedding matirx shape is max words and embeding dim(100)
                                                                 # Where as if we were using an embedding layer with an ML task then input shape would be a 3d tensor of shape (samples,sequence length per sample,desired emedding dimesion of embedding vector itself)        
        #Populate Embedding Matrix in this cell
    wordsnotfoundinembeddingfile = []
    for word,its_index in word_index_dict.items():              #word_index_dict is a dict of words and their assigned indices by the tokenizer on the original corpus
        #if its_index < top_maxwords_corpus:                    #only considering 1st 10K words
            matrix_index = its_index#-1                         #its_index-1 approach may have some logic because word indices start from 1 but matrix first row index should be 0. This is done not just to avoid an error but to avoid a blank 1st row. we do not add 1 to vocab size just to avoid the error. But by this approach we get strange index errors, so i just went back to adding +1 to vocab size in topmaxwords
            embedding_vector = embeddings_index_dict.get(word)  #gives the coefs of each word ( floaiting points are value pair of key the word)
            if embedding_vector is not None:                    #if word from corpus is also found in glove file
                embedding_matrix[matrix_index] = embedding_vector  #then add glove coeffs of that word into emeddings_matrix array
            else:
                print(word)
                wordsnotfoundinembeddingfile.append(word)#if word from corpus is not found in Glove file it will have just have the initialized values(zeros)        
    return embedding_matrix,wordsnotfoundinembeddingfile


# In[ ]:


def data_eda(X):
    classcounts = pd.DataFrame(X['Child Class L1'].value_counts())
    fig = px.bar(classcounts, x=classcounts.index,  y=classcounts['Child Class L1'])
    fig.show()


# In[ ]:


def create_model(sequence_length,
                 embedding_dim,
                 top_maxwords_atcorpuslevel,
                 embedding_matrix,
                 num_child_classes, 
                 filter_sizes,
                 num_filters, 
                 drop,
                 regularize):

# Model Arch 1 : Series of  ConV2D+Pool layers feeding 1 into another befire finally flattening and fed into Dense Layer
# Same # of filters, but varying filter sizes , keeping pool sizes remaining same , Square Filters used

#     top_maxwords_atcorpuslevel = vocabulary_size

    inputs = Input(shape=(sequence_length,) , dtype='int32')#(None, 1680)  

    embedding = Embedding(    input_dim=top_maxwords_atcorpuslevel, #vocabulary_size,
                              output_dim=embedding_dim,
                              weights=[embedding_matrix],trainable=False
                         )   (inputs) #(None, 99, 100)trainable=False(batch_size, sequence_length, output_dim) 

    reshape = Reshape((sequence_length,embedding_dim,1))(embedding) #(None, 99, 100, 1)  
    conv0 = Conv2D(filters = 36,kernel_size=(filter_sizes[0],filter_sizes[0] ), activation='relu', padding='valid')(reshape) #(None, 97, 1, 512) #99, 100, 1 -->#n-f+p +1 = 99-3+1 =97.
    pool0 = MaxPool2D(pool_size=(2,2),strides=(1,1))(conv0)

    conv1 = Conv2D(filters = 36,kernel_size=(filter_sizes[1],filter_sizes[1] ), activation='relu', padding='valid')(pool0) #(None, 97, 1, 512) #99, 100, 1 -->#n-f+p +1 = 99-3+1 =97.
    pool1 = MaxPool2D(pool_size=(2,2),strides=(1,1))(conv1)

    conv2 = Conv2D(filters = 36,kernel_size=(filter_sizes[2],filter_sizes[2] ), activation='relu', padding='valid')(pool1) #(None, 97, 1, 512) #99, 100, 1 -->#n-f+p +1 = 99-3+1 =97.
    pool2 = MaxPool2D(pool_size=(2,2),strides=(1,1))(conv2)


    flatten = Flatten()(pool2)
    output  = Dense(num_child_classes, activation='softmax')(flatten)
    model1 = Model(inputs=inputs, outputs=output)
    
# Model Arch 2 :  ConV2D+Pool layers Merged + falttened and fed into a Dense layer Deep Model
# Same # of filters, but varying filter sizes and  pool sizes , Large rectangular filters relative to corpus are used



    inputs = Input(shape=(sequence_length,) , dtype='int32')#(None, 99)  

    embedding = Embedding(    input_dim=top_maxwords_atcorpuslevel, #vocabulary_size,
                              output_dim=embedding_dim,
                              weights=[embedding_matrix],trainable=False
                              )(inputs) #(None, 99, 100)trainable=False(batch_size, sequence_length, output_dim) 

    merge_layers = []
    reshape = Reshape((sequence_length,embedding_dim,1))(embedding) #(None, 99, 100, 1)  
    conv0 = Conv2D(filters = 36,kernel_size=(filter_sizes[0], embedding_dim), activation='relu', padding='valid',kernel_regularizer = regularizers.l2(0.2))(reshape)#data_format='channels_last'
    conv0 = BatchNormalization()(conv0)
    conv0 = Dropout(drop)(conv0)
    ps = sequence_length - filter_sizes[0] + 1 
    pool0 = MaxPool2D(pool_size=(ps,1),strides=(1,1))(conv0)
    merge_layers.append(pool0)


    conv1 = Conv2D(36,kernel_size=(filter_sizes[1], embedding_dim), activation='relu', padding='valid', kernel_regularizer = regularizers.l2(0.2))(reshape)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(drop)(conv1)
    ps = sequence_length - filter_sizes[1] + 1
    pool1 = MaxPool2D(pool_size=(ps,1),strides=(1,1))(conv1)
    merge_layers.append(pool1)

    conv2 = Conv2D(36,kernel_size=(filter_sizes[2], embedding_dim), activation='relu', padding='valid',kernel_regularizer = regularizers.l2(0.2))(reshape)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(drop)(conv2)
    ps = sequence_length - filter_sizes[2] + 1
    pool2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1,1),strides=(1,1))(conv2)
    merge_layers.append(pool2)


    concatenated_tensor = concatenate([pool0,pool1,pool2],axis=-1)

    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output  = Dense(num_child_classes, activation='softmax')(dropout)

    model2 = Model(inputs=inputs, outputs=output)
    
# Model Arch 3 : Series of  ConV1D+Pool layers feeding 1 into another befire finally flattening and fed into Dense Layer
# Same # of filters, but varying filter sizes , keeping pool sizes remaining same 


    inputs = Input(shape=(sequence_length,) , dtype='int32')#(None, 99)  

    embedding = Embedding(    input_dim=top_maxwords_atcorpuslevel, #vocabulary_size,
                              output_dim=embedding_dim,
                              weights=[embedding_matrix],trainable=False
                              )(inputs) #(None, 99, 100)trainable=False(batch_size, sequence_length, output_dim) 

    #reshape = Reshape((sequence_length,embedding_dim,1))(embedding) #(None, 99, 100, 1)  
    conv1_0 = Conv1D(filters = 36,kernel_size=filter_sizes[0], activation='relu', padding='valid')(embedding) #(None, 97, 1, 512) #99, 100, 1 -->#n-f+p +1 = 99-3+1 =97.
    pool1_0 = MaxPool1D(pool_size=2,strides=1)(conv1_0)

    conv1_1 = Conv1D(filters = 36,kernel_size= filter_sizes[1] , activation='relu', padding='valid')(pool1_0) #(None, 97, 1, 512) #99, 100, 1 -->#n-f+p +1 = 99-3+1 =97.
    pool1_1 = MaxPool1D(pool_size=2,strides=1)(conv1_1)

    conv1_2 = Conv1D(filters = 36,kernel_size= filter_sizes[2] , activation='relu', padding='valid')(pool1_1) #(None, 97, 1, 512) #99, 100, 1 -->#n-f+p +1 = 99-3+1 =97.
    pool1_2 = MaxPool1D(pool_size=2,strides=1)(conv1_2)


    flatten = Flatten()(pool1_2)
    output  = Dense(num_child_classes, activation='softmax')(flatten)
    model3 = Model(inputs=inputs, outputs=output)   

# Model Arch 4 :  ConV1D+Pool layers Merged + falttened and fed into a Dense layer Deep Model
# Same # of filters, but varying filter sizes and  pool sizes , Large rectangular filters relative to corpus are used
    

    inputs = Input(shape=(sequence_length,) , dtype='int32')#(None, 99)  

    embedding = Embedding(    input_dim=top_maxwords_atcorpuslevel, #vocabulary_size,
                              output_dim=embedding_dim,
                              weights=[embedding_matrix],trainable=False
                              )(inputs) #(None, 99, 100)trainable=False(batch_size, sequence_length, output_dim) 

    merge_layers = []
    # reshape = Reshape((sequence_length,embedding_dim,1))(embedding) #(None, 99, 100, 1)  
    conv1_0 = Conv1D(filters = 36,kernel_size = filter_sizes[0] , activation ='relu', padding='valid')(embedding)#data_format='channels_last'
    conv1_0 = BatchNormalization()(conv1_0)
    conv1_0 = Dropout(drop)(conv1_0)
    ps = sequence_length - filter_sizes[0] + 1 
    pool1_0 = MaxPool1D(pool_size=ps,strides=1)(conv1_0)
    merge_layers.append(pool1_0)


    conv1_1 = Conv1D(36,kernel_size=filter_sizes[1], activation='relu', padding='valid', )(embedding)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = Dropout(drop)(conv1_1)
    ps = sequence_length - filter_sizes[1] + 1
    pool1_1 = MaxPool1D(pool_size=ps,strides=1)(conv1_1)
    merge_layers.append(pool1_1)

    conv1_2 = Conv1D(36,kernel_size=filter_sizes[2],  activation='relu', padding='valid')(embedding)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = Dropout(drop)(conv1_2)
    ps = sequence_length - filter_sizes[2] + 1
    pool1_2 = MaxPool1D(pool_size=ps,strides=1)(conv1_2)
    merge_layers.append(pool1_2)


    concatenated_tensor = concatenate([pool1_0,pool1_1,pool1_2],axis=-1)

    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output  = Dense(num_child_classes, activation='softmax')(dropout)

    model4 = Model(inputs=inputs, outputs=output)    
    
    return model1,model2,model3,model4


# In[ ]:


# def plot_training_history(history,metric_train,metric_val):

#     acc         = history.history['accuracy']
#     val_acc     = history.history['val_accuracy']
#     loss        = history.history['loss']
#     val_loss    = history.history['val_loss']

#     epochs      = range(1, len(acc)+ 1 )


#     plt.plot(epochs, acc,     'b', label = 'Training Accuracy'   , color = 'b')
#     plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy' , color = 'r')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Training and Validation Accuracy')

#     plt.legend()
#     plt.figure()

#     plt.plot(epochs, loss,     'b', label = 'Training Loss' , color = 'b')
#     plt.plot(epochs, val_loss, 'r', label = 'Validaton Loss', color = 'g')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')

#     plt.legend()
#     plt.figure()

#     plt.show()


# In[ ]:


def plot_training_history_custom(history,metric_train,metric_val,loss_train,loss_val):

    metric_train_values   = history.history[metric_train]
    metric_val_values     = history.history[metric_val]
    loss_train_values     = history.history[loss_train]
    loss_val_values       = history.history[loss_val]
    epochs         = range(1, len(metric_train_values)+ 1 )
    
    

    label = 'Training' + str(metric_train)
    print(label)
    plt.plot(epochs, metric_train_values, 'b', label = label  ,color = 'b')
    label = 'Val' + str(metric_val)
    plt.plot(epochs, metric_val_values, 'r', label = label,color = 'r')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1-Score')

    plt.legend()
    plt.figure()
    plt.show()
    
    plt.plot(epochs, loss_train_values,     'b', label = 'Training Loss' ,color = 'b')
    plt.plot(epochs, loss_val_values, 'r', label = 'Validaton Loss',color = 'g')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.legend()
    plt.figure()

    plt.show()


# In[ ]:


def get_f1(y_true, y_pred): #taken from old keras source code
    print(y_true.shape, y_pred.shape)
    print(y_true, y_pred)
    true_positives = k.backend.sum(k.backend.round(k.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.backend.sum(k.backend.round(k.backend.clip(y_true, 0, 1)))
    predicted_positives = k.backend.sum(k.backend.round(k.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + k.backend.epsilon())
    recall = true_positives / (possible_positives + k.backend.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+k.backend.epsilon())
    return f1_val


# In[ ]:


import h5py
import numpy as np


# In[ ]:





# <h6> Inside a HDF5 file : There are Groups and Data sets. Groups are like folders/containers and contain datasets,other groups.Datasets are actual data like numpy arrays.
# <!-- Groups work like dictionaries, and datasets work like NumPy arrays <h6 -->
# 

# In[ ]:


def isGroup(obj):
    if isinstance(obj,h5py.Group):
        return True
    
    
    return False

def isDataset(obj):
    if isinstance(obj,h5py.Dataset):
        return True
    return False

#Recursive function : keeps going through the object's Nested Groups until it finds a dataset
def getDatasetsFromGroup(datasets,obj):
    if isGroup(obj):
        for key in obj:
            x = obj[key]
            getDatasetsFromGroup(datasets,x)
    else:
        datasets.append(obj)

        
    
def getWeightsForLayer(layerName, fileName):
    
    weights = []

    with h5py.File(fileName, mode = 'r') as f:
        for key in f:
            if layerName in key:
                obj = f[key]
                datasets = []
                getDatasetsFromGroup(datasets,obj)
#                 weights = np.array()
                for dataset in datasets:
                    w = np.array(dataset)
                    weights.append(w)
                    
    return weights
                       


    


# In[ ]:


def printModelH5Contents(fileName):
    with h5py.File(fileName, mode = 'r') as f:    
            for key in f:
                print(key,f[key])
                o = f[key]
                for key1 in o:
                    print(key1,o[key1])
                    r = o[key1]
                    for key2 in r:
                        print(key2,r[key2])


# In[ ]:


def debug_populated_embeddings_matrix(embeddings_index,embedding_matrix,word_index_dict,n):
#     print('Number of words foud in Embeddingfile',wordsfoundinembeddingfile)
#     blankcount = 0
#     nonblankcount = 0
#     for row in embedding_matrix:
#         if  row[2] == 0.0:
#             blankcount +=1
#         else:
#             nonblankcount +=1 
#     print(blankcount,nonblankcount)
#     count = 0
#     for row in embedding_matrix:
#         if row[0] == 0:
#             count += 1
#     print(count)
    
    l = list(word_index_dict.keys())
    tenth_word = l[n]
    print(tenth_word)
    print('Check for zero : if same embeddings are present in both Embeddings Index and Matrix')
    diff = embeddings_index[tenth_word][0:5] - embedding_matrix[n+1][0:5]
    return diff


# In[ ]:


def example_emb_matrix():
    word_index_dict_temp = {'the' : 1,
                             'to' : 2,
                             'a'  : 3,
                             'and': 4,
                             'in' : 5,
                             'of' : 6,
                             'for': 7,
                             'he' : 8,
                             'on' : 9,
                             'i'  : 10}
    matrix = np.zeros((10,100))   # notice np.zeroes(inner tuple bracket), #type array
    for word,its_index in word_index_dict_temp.items():              #word_index_dict is a dict of words and their assigned indices by the tokenizer on the original corpus
        matrix_index = its_index-1
        print(word,matrix_index)
        #if its_index < top_maxwords_corpus:                     #only considering 1st 10K words
        embedding_vector = embeddings_index_dict.get(word)       #gives the coefs of each word ( floaiting points are value pair of key the word)
        if embedding_vector is not None:                    #if word from corpus is also found in glove file
            matrix[matrix_index] = embedding_vector  #then add glove coeffs of that word into emeddings_matrix array     
    print(matrix)

