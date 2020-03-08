import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args
from mnist import MNIST #Imported via script x = img, y = labels (true or pred) ~train on training set, test on testing set, then check y_pred for 1 hot vec, y_pred_cls for label name 


#Constructing model of search space for params, process called Bayesian Optimization. Model is Gaussian Process
#Gaussian Process estimates how perf varies w hyper-params, req Bayesian optimizer to provide params for search space haven't explored, or one that may inc perf
#Repeat process multiple times, updating Gaussian until ideal set found 


#Define range to eval hyper-params over, for learning_rate: 1e-6 to 1e-2, yet too large, therefore can search for k in 1ek to red lower/upper bound
dim_learning_rate = Real(low = 1e-6, high = 1e-2, prior = 'log-uniform', name = 'learning_rate') #Sample params by log scale
dim_num_dense_layers = Integer(low = 1, high = 5, name = 'num_dense_layers') #atleast 1 and at most 5 dense layer
dim_num_dense_nodes = Integer(low = 5, high = 512, name = 'num_dense_nodes')
dim_activation = Categorical(categories = ['relu', 'sigmoid'], name = 'activation')
dimensions = [dim_learning_rate, dim_num_dense_layers, dim_num_dense_nodes, dim_activation]
default_params = [1e-5, 1, 16, 'relu']

#Create parent dir to log/view res param comb, each one stored in subdir
def log_dir_name(learning_rate, num_dense_layers, num_dense_nodes, activation): #Tensorboard provides data visualization
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"
    log_dir = s.format(learning_rate, num_dense_layers, num_dense_nodes, activation)
    return log_dir

#Data setup, input_data comes from import 
data = MNIST(data_dir="data/MNIST/")
print("Training Set: {}".format(data.num_train))
print("Test Set: {}".format(data.num_val))
print("Validation Set: {}".format(data.num_test))
    
validation_data = (data.x_val, data.y_val_cls) #test data for Gaussian Process

#Helper func plot img
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(data.img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)        
        ax.set_xticks([])
        ax.set_yticks([])   
    plt.show()

#Helper func plot misclassified img test set
def plot_example_errors(cls_pred):
    incorrect = (cls_pred != data.test.cls)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

#Set up img 
images = data.x_train[0:9]
cls_true = data.y_train_cls[0:9]
plot_images(images = images, cls_true = cls_true)

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10
img_shape_full = (img_size, img_size, 1)
#CNN creation based on hyper params
def create_model(learning_rate, num_dense_layers, num_dense_nodes, activation):
    model = Sequential()
    model.add(InputLayer(input_shape=(img_size_flat,))) #input layer is tuple of img size and number of images (unlimited)
    model.add(Reshape(img_shape_full)) #reshape for CNN
    model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation = activation, name='layer_conv1')) #want to optimization passed in activ func
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation = activation, name='layer_conv2'))
    model.add(MaxPooling2D(pool_size=2, strides=2)) #Flatten CNN output for input to fully connected layer
    model.add(Flatten())
    for i in range(num_dense_layers): #Add dense layers 
        name = 'layer_dense_{0}'.format(i+1)
        model.add(Dense(num_dense_nodes, activation = activation, name = name))
    model.add(Dense(num_classes, activation='softmax')) #Add final layer w/ output equal to classes
    optimizer = Adam(lr = learning_rate)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model
    
path_best_model = 'best_model.keras' #defn ideal hyper param model
best_accuracy = 0.0
    
#Defn func to create & train NN, eval perf on test set, then return fitness/objective val, neg classification accuracy on test set, neg as minimized
#Skopt uses decorate with params to exec func fitness directly w list-input as defn earlier, skopt converts list to required params for func to use
#Gaussian Model gets modified and updated via decorator 
#Determining fitness val as input to Bayesian optimization 
@use_named_args(dimensions = dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation):
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation:', activation)
    print()
    
    model = create_model(learning_rate = learning_rate, num_dense_layers = num_dense_layers, num_dense_nodes = num_dense_nodes, activation = activation) #create NN
    log_dir = log_dir_name(learning_rate, num_dense_layers, num_dense_nodes, activation)
    #callback func are set func appl at training steps, used to get internal view of var states & statistics
    callback_log = TensorBoard(log_dir = log_dir, histogram_freq = 0, batch_size = 32, write_graph = True, write_grads = False, write_images = False) #won't compute histograms for data, set to zero
    #train model
    history = model.fit(x = data.x_train, y = data.y_train_cls, epochs = 3, batch_size = 128, validation_data = validation_data, callbacks = [callback_log])
    accuracy = history.history['val_acc'][-1]
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    global best_accuracy #global allows modification to var outside of scope 
    if accuracy > best_accuracy:
        model.save(path_best_model)
        best_accuracy = accuracy 
    del model
    K.clear_session()
    return -accuracy

#Run hyper param optimization
fitness(x = default_params) #acq func defn how to find new hyper params from internal model Bayesian optimizer, n_calls is num of iterations
search_result = gp_minimize(func = fitness, dimensions = dimensions, acq_func = 'EI', n_calls = 40, x0 = default_params)

#Load and test on data
model = load_model(path_best_model)
result = model.evaluate(x = data.x_test, y = data.y_test_cls)
images = data.x_test[0:9]
y_pred = model.predict(x = images)