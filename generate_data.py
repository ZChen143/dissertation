import dcgan
import numpy as np
import tensorflow as tf

#
for i in range(4):  # memory may not be enough
CLASS = 3
generator = dcgan.DCGAN.make_generator_model()
generator.load_weights('./model/CLASS' + str(CLASS) + '.h5')
generation = generator(tf.random.normal([10000, 100]), training=False)
np.save('./dataset/generated_tracks_CLASS' + str(CLASS), generation)
# dcgan.plt_tracks(np.load('./dataset/andi_train_tracks.npy')[0:5])
dcgan.plt_tracks(generation[0:5])


tracks = []
labels = []
for i in range(4):
    CLASS = i
    tracks.append(np.load('./dataset/generated_tracks_CLASS' + str(CLASS) + '.npy'))
    labels.append(np.zeros(10000) + i)
tracks = np.array(tracks).reshape(40000, 999, 2)
labels = np.array(labels).reshape(40000)

np.save('./dataset/generated_tracks_all', tracks)
np.save('./dataset/generated_labels', labels)

for i in range(4):  # memory may not be enough
CLASS = 2
generator = dcgan.DCGAN.make_generator_model()
generator.load_weights('./model/CLASS' + str(CLASS) + '_exp.h5')
generation = generator(tf.random.normal([10000, 100]), training=False)
np.save('./dataset/generated_tracks_CLASS_exp' + str(CLASS), generation)
# dcgan.plt_tracks(np.load('./dataset/andi_train_tracks.npy')[0:5])
dcgan.plt_tracks(generation[0:5])

