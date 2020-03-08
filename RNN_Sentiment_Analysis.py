import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import imdb

#Load x-vals as snippets of text, y-val will be sentiment prediction (pos/neg review)
imdb.maybe_download_and_extract()
x_train_text, y_train = imdb.load_data(train = True)
x_test_text, y_test = imdb.load_data(train = False)
print("Train-set size: ", len(x_train_text))
print("Test-set size:  ", len(x_test_text))
data_text = x_train_text + x_test_text
print(x_train_text[1])
print(y_train[1])