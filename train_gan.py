import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing

import dcgan

tf.config.set_soft_device_placement(True)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

model1 = dcgan.DCGAN()

andi_train_x = np.load('./dataset/andi_train_tracks.npy')
andi_train_y = np.load('./dataset/andi_train_labels.npy')

# swimmers = pd.read_csv('./dataset/swimmers5um.csv')
# # swimmers2 = pd.read_csv('swimmers2um.csv')
# # swimmers5 = pd.read_csv('swimmers5um.csv')
#
# col = swimmers.iloc[0, :].size
# row = swimmers.iloc[:, 0].size
# print('columns: %d \nrows: %d' % (col, row))
#
# list_dataset = []
#
# for i in range(1, int((col + 1) / 2)):
#     X = [preprocessing.minmax_scale(swimmers.iloc[:, i * 2 - 1])*2-1,
#          preprocessing.minmax_scale(swimmers.iloc[:, i * 2])*2-1]  # clear X
#     list_dataset.append(np.array(X).T)
#
# train_dataset = np.array(list_dataset).reshape(5, 999, 2)
#
# CLASS = 2
# model1 = dcgan.DCGAN()
# g_loss, d_loss = model1.train(tf.data.Dataset.from_tensors(train_dataset), 10000)
# path = './model/CLASS' + str(CLASS) + '_exp.h5'
# model1.generator.save(path)
# plt.plot(g_loss)
# plt.plot(d_loss)
# plt.show()
# plt.close()



for j in range(4):
    CLASS = j
    train_dataset = andi_train_x[j * 5: (j + 1) * 5]
    model1 = dcgan.DCGAN()
    g_loss, d_loss = model1.train(tf.data.Dataset.from_tensors(train_dataset), 10000)
    path = './model/CLASS' + str(CLASS) + '.h5'
    model1.generator.save(path)

    plt.plot(g_loss)
    plt.plot(d_loss)
    plt.show()
    plt.close()
