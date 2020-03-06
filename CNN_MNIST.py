import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import math
from mnist import MNIST


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
        layer = tf.nn.max_pool(value = layer,
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
num_filters1 = 16

#Conv Layer 2; Take sliding sum by applying all one filter from each channel (16 available) 
filter_size2 = 5 #Channels output by conv; each one has filter set of diff size in next conv; determines next amt channels
num_filters2 = 36 #Repeated 36 times to cover all channels, which are depth of layer 


#FC (defn neurons)
fc_size = 128

#Data import and setup
data = MNIST(data_dir="data/MNIST/")

print("Training Set: {}".format(data.num_train))
print("Test Set: {}".format(data.num_val))
print("Validation Set: {}".format(data.num_test))

#Input data setup (one channel for greyscale)
img_size = data.img_size
img_size_flat = data.img_size_flat
img_shape = data.img_shape
num_channels = data.num_channels #one img start, one channel
num_classes = data.num_classes

#Plot first nine images and true classes
inputImg = data.x_test[0:9]
trueCls = data.y_test_cls[0:9]
plot_images(inputImg, trueCls)

#Input to CNN via placeholder var
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x') #arbitrary num labels, each input to conv
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels]) #encode as 4-dim tensor for input 
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true') #each label vec of length num_classes (10)
y_true_cls = tf.argmax(y_true, axis=1) #axis = 1 defns classes across cols and data across rows 

#Create CNN layer 1
layer_conv1, weights_conv1 = new_conv_layer(input = x_image, 
                                            num_input_channels = num_channels,
                                            filter_size = filter_size1,
                                            num_filters = num_filters1,
                                            use_pooling = True)
print(layer_conv1) #Output of layer is shape: ?, 14, 14, 16 ~ unknown num img are 14x14, prod 16 channels

#Create CNN layer 2
layer_conv2, weights_conv2 = new_conv_layer(input = layer_conv1, 
                                            num_input_channels = num_filters1,
                                            filter_size = filter_size2,
                                            num_filters = num_filters2,
                                            use_pooling = True)
print(layer_conv2)#Output of layer is shape: ?, 7, 7, 36 ~ unknown num img are 7x7, prod 36 channels

#Flatten, pass into FCN, determine predicted class 
layer_flat, num_features = flatten_layer(layer_conv2)
print(layer_flat) #7x7 img, 36 var copies = 1764 features for FCN

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs = num_features,
                         num_outputs = fc_size,
                         use_relu = True)
layer_fc2 = new_fc_layer(input=layer_fc1, 
                         num_inputs = fc_size,
                         num_outputs = num_classes,
                         use_relu = False)
y_pred = tf.nn.softmax(layer_fc2) #linearize all nums to [0, 1]
y_pred_cls = tf.argmax(y_pred, axis = 1) #index of largest element ~ predicted class

#Check true against pred via cross-entropy cost func
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels = y_true)
cost = tf.reduce_mean(cross_entropy)

#Set up optimizier by adding to comp graph, adv one for grad desc
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Setting up TF
session = tf.compat.v1.Session()
session.run(tf.global_variables_initializer())

#Setting up train/test for model
train_batch_size = 64
total_iterations = 0

#Func for running optimizer on batch size by iterations; training model 
def optimize(num_iterations):
    global total_iterations
    start_time = time.time()
    for i in range(total_iterations, total_iterations + num_iterations):
        #Batch of img and true labels
        x_batch_img, y_batch_true_cls, _ = data.random_batch(batch_size = train_batch_size)
        #Dict for placeholder var names
        feed_dict_train = {x: x_batch_img, y_true: y_batch_true_cls}
        session.run(optimizer, feed_dict = feed_dict_train)
        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict = feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))
    total_iterations += num_iterations
    end_time = time.time()
    time_diff = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))
       
    
#Confusion matrix shows differences between test and training data to show connections; helpful to compare models 
#Rows rep predicted, cols rep known truth; true pos in left corner, true neg in right corner for a 4x4 matrix
#true pos is data model correctly guessed  true, true neg is data model correctly guessed false
#bot left & bot right are false neg & pos resp; false neg = truth is cat A, but algo pred cat B
#false pos = truth is cat B, but algo pred cat A, more relevant in context of data; regardless are misclassifications
#Diagonal of conf mat always correct predictions
def plot_confusion_matrix(cls_pred):
    # cls_pred arr pred class num for img 

    # Get truth
    cls_true = data.y_test_cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix
    print(cm)

    # Plot the confusion matrix
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()    
    
test_batch_size = 256
#Func to show performance for testing model in batches
def print_test_accuracy():
    num_test = data.num_test #img test set; data refers to MNIST set
    cls_pred = np.zeros(shape = num_test, dtype = np.int) #filled in batches
    i = 0 #starting ind batch = i, ending = j
    while i < num_test:
        j = min(i + test_batch_size, num_test) #check if near end or a batch
        images = data.x_test[i:j, : ] #img b/w i & j, x_test, y_test custom defn as MNIST remove
        labels = data.y_test[i:j, : ] #rep same as data.test.images & data.test.labels
        feed_dict = {x:images, y_true:labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict = feed_dict) #populate arr of pred for batch
        i = j
    cls_true = data.y_test_cls 
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum() #false = 0, true = 1, count right pred
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    print("Confusion Matrix:")
    plot_confusion_matrix(cls_pred=cls_pred)
    
optimize(num_iterations = 1000)
print_test_accuracy()

#Func plot weights 
def plot_conv_weights(weights, input_channel = 0):
    #assume weights are 4-dim tensors 
    w = session.run(weights) #retrieve TF weights
    w_min = np.min(w) #retrieve min/max for normalization by matplotlib to range 0 - 255 for display
    w_max = np.max(w)
    num_filters = w.shape[3] #given tensor, at index 3, num channels 
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids) #axes np arr of dim; must flatten to iterate (single list)
    
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = w[:, :, input_channel, i] #get all img, and iterate by channel
            ax.imshow(img, vmin = w_min, vmax = w_max, interpolation = 'nearest', cmap = 'seismic')
            
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

#Func plot output conv layer
def plot_conv_layer(layer, image):
    feed_dict = {x: [image]} #feed-dict containing single img
    values = session.run(layer, feed_dict = feed_dict) #find output after running
    num_filters = values.shape[3] #filters used in layer
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    
    #output img per each filter applied
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = values[0, :, :, i] #ith channel
            ax.imshow(img, interpolation = 'nearest', cmap = 'binary') #nearest interpolation does not insert addl pixels for diff in res b/w output and given img
            #cmap is colour map, binary is all white to black
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

#Func to plot input img
def plot_single_img(image):
    plt.imshow(image.reshape(img_shape), interpolation = 'nearest', cmap = 'binary')
    plt.show()
    
#See output of conv layers and weights
image1 = data.x_test[0]
plot_single_img(image1)
image2 = data.x_test[3]
plot_single_img(image2)

plot_conv_weights(weights = weights_conv1) #retrieved from conv layer 1 instantiation, red is pos weights, blue is neg weights 
plot_conv_layer(layer = layer_conv1, image = image1) #colours weights result of sliding across that input img 
plot_conv_weights(weights = weights_conv2, input_channel = 0) #there are 15 more input channels, realise each one prod 36 filters
plot_conv_layer(layer = layer_conv2, image = image2) #for greater conv, model finds even more minute patterns, reaches point where numbers are identifiable by human eye



session.close()
    

    
    
