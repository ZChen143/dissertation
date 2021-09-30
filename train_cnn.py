import dcgan
import numpy as np
import tensorflow as tf

from models import classification_model_2d
from tensorflow.keras.optimizers import Adam

x = np.load('./dataset/generated_tracks_all.npy')
y = np.load('./dataset/generated_labels.npy')
shuffle_indices = np.random.permutation(int(len(y)))

indices_train = int(len(y) * 0.8)

train_x = x[shuffle_indices[0:indices_train]]
train_y = y[shuffle_indices[0:indices_train]]
val_x = x[shuffle_indices[indices_train:]]
val_y = y[shuffle_indices[indices_train:]]

model = classification_model_2d()
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_x, train_y, validation_data=(val_x, val_y),
                    use_multiprocessing=True, workers=8)

model.save('./model/cnn.h5')
