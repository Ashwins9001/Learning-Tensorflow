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

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
img_shape_full = (img_size, img_size, 1)

model = Sequential()

model.add(InputLayer(input_shape=(img_size_flat,)))

model.add(Reshape(img_shape_full))

model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                     activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                     activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten())

for i in range(5):
    name = 'layer_dense_{0}'.format(i+1)

    model.add(Dense(1024,
                    activation='relu',
                        name=name))

model.add(Dense(1024, activation='softmax'))
    
optimizer = Adam(lr=2)
    
model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    