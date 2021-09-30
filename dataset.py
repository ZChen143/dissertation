import numpy as np
import andi
import dcgan
import numpy as np
from andi import diffusion_models
from sklearn import preprocessing


num_of_each_cls = 1000
dimensions = 2
length = 999
AD = andi.andi_datasets()
diff_mod = diffusion_models.twoD()
tracks = []
labels = []

dataset = AD.create_dataset(T=length, N=num_of_each_cls, exponents=[0.5, 0.9, 1.4, 1.8], models=[2], dimension=2)

for i in range(num_of_each_cls * 4):
    tracks.append(preprocessing.minmax_scale(dataset[i, 2:].reshape(2, 999).T)*2-1)
tracks = np.array(tracks)

for i in range(4):
    labels.append(np.zeros(num_of_each_cls) + i)
labels = np.array(labels).reshape(num_of_each_cls * 4)

np.save('./dataset/andi_test_tracks', tracks)
np.save('./dataset/andi_test_labels', labels)

# dataset = AD.create_dataset(T=length, N=5, exponents=[0.5, 0.9, 1.4, 1.8], models=[2], dimension=2)
#
# for i in range(20):
#     tracks.append(preprocessing.minmax_scale(dataset[i, 2:].reshape(2, 999).T)*2-1)
# tracks = np.array(tracks)
#
# np.save('./dataset/andi_train_tracks', tracks)
