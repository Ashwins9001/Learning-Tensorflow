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
    
print("works")