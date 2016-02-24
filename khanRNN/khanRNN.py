
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from six.moves import range
import random,cPickle
import numpy as np
import random, sys


# In[2]:

data = np.random.randn(100, 1, 101) # nb_samples=100, timesteps=1, input_dim=101)
label = np.zeros([100, 6, 1])
label[0:50, :, :] = 0.0 
label[50:100, :, :] = 1.0


# In[3]:

## splite data
(X_train,X_val) = (data[0:80], data[80:])   # training data
(Y_train,Y_val) = (label[0:80], label[80:]) # val data
best_accuracy = 0.0


# In[4]:

## train and val
print('Train')
nb_epoch = 8
batch_size = 20
nb_class = 2
for e in range(nb_epoch):
    print('epoch', e)
    batch_num = len(Y_train)/batch_size
    progbar = generic_utils.Progbar(X_train.shape[0])
    for i in range(batch_num):
        x = X_train[i*batch_size:(i+1)*batch_size]
        y = Y_train[i*batch_size:(i+1)*batch_size]
        loss,accuracy = model.train_on_batch(x, y, accuracy=True) # train
        progbar.add(batch_size, values=[("train loss", loss),("train accuracy:", accuracy)] )

