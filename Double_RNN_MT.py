#Use encoder-decoder network with two separately trained RNNs, first one recovers sentiment from a sentence (thought vec) written in another language
#Pass sentiment as single vec to input to GRUs of second RNN, which maps thought vec to next word and forms a translate sentence
#Prob func used in second RNN to determine best translation-word as per an alignment func
#Each GRU designed with update & reset gates, reset determines if previous information should be passed alongside current input, and update determines if we pass prev time step or updated curr time step
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import os
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import europarl #file to dl & load data

language_code='da' #english-to-danish set two millions of lang-pairs
mark_start = 'ssss ' #start & end pt for decoder that are unlikely to be words in sentence
mark_end = ' eeee'
europarl.maybe_download_and_extract(language_code=language_code)
data_src = europarl.load_data(english=False, #src lang (dan)
                              language_code=language_code)
data_dest = europarl.load_data(english=True, #dest lang (eng)
                               language_code=language_code,
                               start=mark_start,
                               end=mark_end)
idx = 2
data_src[idx]
data_dest[idx]
