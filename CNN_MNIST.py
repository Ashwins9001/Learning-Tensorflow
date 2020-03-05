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

def new_weights(shape): #Apply random weights across some matrix w dim defn by shape
    return tf.Variable(tf.truncated_normal(shape, stddev=0.5))

def new_biases(length): #Create arr size length, fill const bias
    return tf.Variable(tf.constant(0.05, shape=[length]))

#Conv layer setup: input - 4-dim tensor of img num, x dim, y dim, channels; output - 4-dim tensor of img num, x dim / 2, y dim / 2, channels by each conv filter
#Assuming 2x2 pooling used to reduce size
#Conv filter slid across and weighted sum reps similarity b/w it and class; provides score to alter weights
#Pooling takes local max val to red img and maintain information; ReLu prevents neg nums to keep NN from going to extremes
#FCN then red to classes, provides votes; highest one is predicted cls
#CNN trained on error b/w test img passed in and pred; backprop
def new_conv_layer(input, #Prev layer
                   num_input_channels, #Channels prev layer; prod set of filters per EACH channel in next conv
                   filter_size, 
                   num_filters, 
                   use_pooling=True):
    #Init random filters & biases
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weightsPerFilter = new_weights(shape)
    biases = new_biases(length=num_filters)
    
    #Stride first & last params one pixel to prevent filter from exceeding img area; moves 1 across x and y
    #Padding implies zeros padded to maintain output size
    layer=tf.nn.conv2d(input = input, 
                       filter = weightsPerFilter,
                       strides = [1, 1, 1, 1],
                       padding = 'SAME')
    layer += biases #Added to each channel 
    if use_pooling: #strides move 2 across x & y. defn 2x2 window in ksize
        layer = tf.nn.maxpool(value = layer,
                              ksize = [1, 2, 2, 1],
                              strides = [1, 2, 2, 1],
                              padding = 'SAME')
    layer = tf.nn.relu(layer) #Relu executed before max_pool, but: relu(max_pool(x)) == max_pool(relu(x)); saves operations and inc efficiency
    return layer, weightsPerFilter

#Require 4-dim tensor layer to be flattened to 2-dim tensor for FCN
def flatten_layer(layer):
    layer_shape = layer.get_shape() #Get dimensionality; [num_img, img_height, img_width, num_channels]
    #Use features to detect patterns and differentiating factors b/w classes; num of feat is img_height * img_width * num_channels
    #Channels input to next conv, possible patterns
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features]) #calc size first dim, second set to num for init; [num_img, img_height * img_width * num_channels]
    return layer_flat, num_features
    
#Post flatenning, require FCN: [num_images, num_inputs] -> [num_images, num_outputs]
def new_fc_layer(input,
                 num_inputs,
                 num_outputs,
                 use_relu = True): #Define weights, biases for training
    weights = new_weights(shape = [num_inputs, num_outputs])
    biases = new_biases(length = num_outputs)
    layer = tf.matmul(input, weights) + biases #[num_inputs] * [num_inputs, num_outputs] = num_outputs
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer
    
#Conv Layer 1
filter_size1 = 5
num_filters = 16

#Conv Layer 2; Take sliding sum by applying all one filter from each channel (16 available) 
filter_size2 = 5 #Channels output by conv; each one has filter set of diff size in next conv; determines next amt channels
num_filters = 36 #Repeated 36 times to cover all channels, which are depth of layer 


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


