import pandas as pd
import csv
import numpy as np

from random import random
from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Activation
from keras.layers.recurrent import LSTM

import matplotlib
matplotlib.use('Agg')

from tsne import tsne

import sklearn.cluster as sk

import matplotlib.pyplot as Plot

# data = pd.read_csv('khan_all_clean.csv')
#
# problems = pd.DataFrame()
# problems.insert(0, 'false', np.zeros(len(data), dtype=np.int))
# problems.insert(1, 'true', np.zeros(len(data), dtype=np.int))
#
# skill_col = data['problem_exercise']
# skills = set()
# for skill in skill_col:
#     if skill not in skills:
#         skills.add(skill)
#
# for skill in skills:
#     problems[skill] = 0
#
# added_rows = 0
# past_student = None
# new = 0
#
# new_row = np.zeros(len(skills)+2, dtype=np.int)
#
# for i, row in enumerate(data.values):
#     if row[0] != past_student:
#         new += 1
#         problems.loc[i+added_rows] = new_row
#         problems.loc[len(problems)] = new_row
#         added_rows += 1
#         past_student = row[0]
#     if row[1]:
#         problems['true'].iloc[i + added_rows] = 1
#     else:
#         problems['false'].iloc[i + added_rows] = 1
#     for skill in skills:
#         if row[4] == skill:
#             problems[skill].iloc[i + added_rows] = 1
#
# labels = problems.iloc[1:].copy()
# labels.loc[len(labels)] = np.zeros(len(skills)+2, dtype=np.int)
#
# def _load_data(data, n_prev = 100):
#     """
#     data should be pd.DataFrame()
#     """
#     docX, docY = [], []
#     for i in range(0, len(data)-n_prev, 100):
#         docX.append(data.iloc[i:i+n_prev-1].as_matrix())
#         docY.append(data.iloc[i+1:i+n_prev].as_matrix())
#     alsX = np.array(docX)
#     alsY = np.array(docY)
#
#     return alsX, alsY
#
# def train_test_split(data):
#     """
#     This just splits data to training and testing parts
#     """
#     ntrn = int(round(len(data) * 0.8))
#
#     train, train_labels = _load_data(data.iloc[0:ntrn])
#     test, test_labels = _load_data(data.iloc[ntrn:])
#
#     return (train, train_labels), (test, test_labels)
#
# (train, train_labels), (test, test_labels) = train_test_split(problems)
#
# print train.shape
# print train.shape
#
# in_neurons = 559
# out_neurons = 559
# hidden_neurons = 256
#
# model = Sequential()
# model.add(LSTM(output_dim=hidden_neurons, input_dim=in_neurons, return_sequences=True))
# model.add(TimeDistributedDense(output_dim=out_neurons, input_dim=hidden_neurons))
# model.add(Activation("linear"))
# model.compile(loss="mean_squared_error", optimizer="rmsprop")
#
# model.fit(train, train_labels, batch_size=150, nb_epoch=5)
#
# model2 = Sequential()
# model2.add(LSTM(output_dim=hidden_neurons, input_dim=in_neurons, return_sequences=True, weights=model.layers[0].get_weights()))
# model2.add(Activation('tanh'))
# model2.compile(loss="mean_squared_error", optimizer="rmsprop")
#
# all_students = dict()
# timesteps = []
# index = 0
# for row in problems.values:
#     if row[0] == 0 and row[1] == 0:
#         timesteps = []
#         all_students[index+1] = timesteps
#         index += 1
#     else:
#         all_students[index].append(row)
#
#
# predictions = dict()
# first = []
#
# for index in all_students:
#     for row in all_students[index]:
#         if index not in predictions:
#             first = []
#             predictions[index] = first
#         else:
#             temp = np.array([[row]])
#             predict = model2.predict(temp)
#             # Check for correct hidden output state
# #             print predict.shape
#             predictions[index].append(predict)
#
# all_timesteps = []
# for student in predictions:
#     if predictions[student]:
#         for hiddens in predictions[student]:
#             all_timesteps.append(hiddens[0][0])
#
# np.savetxt("all_data.txt", np.array(all_timesteps))
from sklearn import manifold
import time
start_time = time.time()

X = np.loadtxt("just10.txt")
m = manifold.TSNE()
Y = m.fit_transform(X)
print Y[0]

# Plot.axis([-30,30,-30,30])
#
# colors = ['red','green','blue','yellow','orange']
# # plt.figure(figsize=(10,10))
#
# first = Y[::99]
# # print len(first)
#
# kmeans = sk.KMeans(n_clusters = 5)
# kmeans.fit(first)
#
# memb = np.array(kmeans.labels_)
#
# for timestep in range(0,99):
#     points = Y[timestep::99]
#     for i in range(5):
#         ind = np.where(memb==i)
#         Plot.scatter(points[ind,0], points[ind,1], 100, colors[i], alpha = 0.5)
#     Plot.savefig(str(timestep) + '.png')
#     Plot.cla()
#
# print("--- %s seconds ---" % (time.time() - start_time))
