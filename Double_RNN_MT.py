#Use encoder-decoder network with two RNNs connected and trained end-to-end, first one recovers sentiment from a sentence (thought vec) written in another language
#Pass sentiment as single vec to input to GRUs of second RNN, which maps thought vec to next word and forms a translate sentence
#Align index of both encoder & decoder to match up for MT 
#Train model on frequently used words of both lang, and allow it to use prob func to determine best word-alignment, in a sense it becomes a RNN search prob
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
print(data_src[idx]) #Some data contains errors
print(data_dest[idx])

#Tokenizer creation to map words to ints, then to vecs (embedding-layer)
max_words = 10000 #max for each lang
class TokenizerWrap(Tokenizer): #self used each time for sep instance of Tokenizer, change each's prop
    #Add func to Tokenizer class in Keras by wrapping it
    def __init__(self, texts, padding, reverse = False, num_words = None): #on init, setup list of tokens per seq
        #padding = post or pre, texts = word strings, num_words = max
        Tokenizer.__init__(self, max_words)
        self.fit_on_texts(texts) #create vocab (int) from txt
        self.index_to_word = dict(zip(self.word_index.values(), self.word_index.keys())) #defn key-val pairs for each word to a token
        self.tokens = self.texts_to_sequences(texts) #create list tokens per each seq, diff length
        if reverse:
            self.tokens = [list(reversed(x)) for x in self.tokens]
            truncating = 'pre' #reversed so truncate beginning, but really end of seq
        else:
            truncating = 'post' #not rev so truncate normally 
        self.tokens_per_seq = [len(x) for x in self.tokens] #tokens per each seq
        
        self.max_tokens = np.mean(self.tokens_per_seq) + 2 * np.std(self.tokens_per_seq)
        self.max_tokens = int(self.max_tokens)
        self.padded_tokens = pad_sequences(self.tokens, maxlen = self.max_tokens, padding = padding, truncating = truncating) #pad token length PER each seq
        
    def token_to_word(self, tokens):
        #convert tokens-list to str
        words = [self.index_to_word[token] for token in tokens if token != 0]
        text = " ".join(words)
        return text
    def text_to_tokens(self, text, reverse = False, padding = False):
        #convert str to tokens w opt pad & rev 
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)
        if reverse:
            tokens = np.flip(tokens, axis = 1)
            truncating = 'pre'
        else:
            truncating = 'post'
        if padding:
            tokens = pad_sequences(tokens, maxlen = self.max_tokens, padding = 'pre', truncating = truncating)
            
#Set up tokenizer for src & dest langs, last words seen by encoder match with first words seen by decoder, improve performance
tokenizer_src = TokenizerWrap(texts = data_src, padding = 'pre', reverse = True, max_words)
tokenizer_dest = TokenizerWrap(texts = data_dest, padding = 'post', reverse = False, max_words)