# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:39:33 2020

@author: Owner
"""

from tensorflow.examples.tutorials.mnist import input_data
"Convert to one-hot encoding to easily pick out number via index"
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
print("Training Set: {} ".format(len(data.train.labels)))
print("Test Set: {} ".format(len(data.test.labels)))
print("Validation Set: {} ".format(len(data.validation.labels)))