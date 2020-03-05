import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9    
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)    
        ax.set_xticks([])
        ax.set_yticks([])   
    plt.show()

#Conv Layer 1
filter_size1 = 5
num_filters = 16

#Conv Layer 2; Take sliding sum by applying all one filter from each channel (16 available) 
#Repeated 36 times to cover all filters 
filter_size2 = 5
num_filters = 36

#FC (defn neurons)
fc_size = 128

#Data import and setup
from tensorflow.examples.tutorials.mnist import input_data 
data = input_data.read_data_sets('data/MNIST', one_hot=True)

print("Training Set: {}".format(len(data.train.labels)))
print("Test Set: {}".format(len(data.test.labels)))
print("Validation Set: {}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)

#Input data setup (one channel for greyscale)
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

#Plot first nine images and true classes
inputImg = data.test.images[0:9]
trueCls = data.test.cls[0:9]
plot_images(inputImg, trueCls)

