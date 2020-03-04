import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data

"Convert to one-hot encoding to easily pick out number via index"
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
print("Training Set: {} ".format(len(data.train.labels)))
print("Test Set: {} ".format(len(data.test.labels)))
print("Validation Set: {} ".format(len(data.validation.labels)))

"Return index of maximum val, one hot therefore return num itself per each ROW"
"Array of classes for resp images"
data.test.cls = np.array([label.argmax() for label in data.test.labels])

"Defn MNIST img size, flatten to 1D, create tuple, defn possible nums (classes)"
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10

"Helper func to plot img; nine in a 3x3 grid"
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    "Enumerate to track obj and index, flatten rows, cols to loop over all nine"
    for i, ax in enumerate(axes.flat):
        "Reshape to retrieve img dims"
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
        
images_data = data.test.images[0:9]
"Set true classes equal to this, taken from test set, for no prediction just show"
cls_true_data = data.test.cls[0:9]
plot_images(images_data, cls_true_data)
        
"Defn placeholder var that are mapped as input img to comp graph"
"A tensor (multidim vec/matrix) with float data-type 32 bits"
"None defn arbitrary rows with cols img_size_flat, thus each row holds img"
x = tf.placeholder(tf.float32, [None, img_size_flat])
"One-hot encoding for true class"
y_true = tf.placeholder(tf.float32, [None, num_classes])
"True number class"
y_true_cls = tf.placeholder(tf.int64, [None])

"Defn var optimization"
"Weights defn linear func steepness, for small changes of that var, large effect on output"
"Class prediction per pixel"
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
"Biases defn pre-existing cond that affect output"
"Possible test set is predisposed to guessing a small set of nums"
biases = tf.Variable(tf.zeros([num_classes]))

"Defn model which returns [num_images, num_classes] given x = [num_images, img_size_flat] and weights = [img_size_flat, num_classes] thus eliminated by matrix multiplication
"Class predictions per each img"
"ith row, jth col reps likelihood of jth class matching ith img"
logits = tf.matmul(x, weights) + biases