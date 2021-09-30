import matplotlib.pyplot as plt
import numpy as np
import pandas
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from models import classification_model_2d

model = classification_model_2d()
model.load_weights('./model/cnn.h5')

x = np.load('./dataset/andi_test_tracks.npy')
y = np.load('./dataset/andi_test_labels.npy')
# shuffle_indices = np.random.permutation(int(len(y)))
#
# test_x = x[shuffle_indices]
# test_y = y[shuffle_indices]

result = model.predict(x)

pre = []
for i in range(len(result)):
    pre.append(result[i].argmax())
pre = np.array(pre)
heatmap(pandas.DataFrame(confusion_matrix(pre, y)), annot=True)
plt.show()
