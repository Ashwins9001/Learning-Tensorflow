import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import imdb

#Load x-vals as snippets of text, y-val will be sentiment prediction (pos/neg review)
imdb.maybe_download_and_extract()
x_train_text, y_train = imdb.load_data(train = True)
x_test_text, y_test = imdb.load_data(train = False)
print("Train-set size: ", len(x_train_text)) #each item is a passage 
print("Test-set size:  ", len(x_test_text))
data_text = x_train_text + x_test_text
print(x_train_text[1])
print(y_train)

#Tokenizer: used to convert words in text strings to nums for NN proc
#Instruct it to take x most popular words, it removes grammar & punc to form list ~ fitting to set
num_words_parse = 10000 #tokenizer will create encoding for each of these 10000 words 
tokenizer = Tokenizer(num_words_parse)
tokenizer.fit_on_texts(data_text) #find words from test & training data
if num_words_parse is None: #Scan entire vocab
    num_words_parse = len(tokenizer.word_index)
print(tokenizer.word_index) #determines int-mapping for each word

#Convert words in test/training set to token form where each word encoded to int 
x_train_tokens = tokenizer.texts_to_sequences(x_train_text)
print(np.array(x_train_tokens[1]))
x_test_tokens = tokenizer.texts_to_sequences(x_test_text)

#Require seq to have same length, RNN can take arbitrary length seq in, yet mem wastage if longest seq used
#Either ensure all seq of data-set are same length, or write custom data-gen that ensures seq eq length w/ in batch
#Use avg seq length to cover most, truncate long, pad short
#Length token is num words in a seq, there can be repetitions as each word just replaced w int
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens] #arr of token lengths for each text seq
num_tokens = np.array(num_tokens)
print(np.mean(num_tokens)) #mean token length : 221.76
print(np.max(num_tokens)) #max token length : 2209
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens) #max is mean plus two std dev
max_tokens = int(max_tokens)
print(max_tokens) #max allowable tokens : 537

#Order of padding & truncating matters, zeros added for pad, integers thrown away for trunc
#Pre or post = throw away FIRST or LAST and pad BEG or END respectively
pad = 'pre'
x_train_pad = pad_sequences(x_train_tokens, maxlen = max_tokens, padding = pad, truncating = pad)
x_test_pad = pad_sequences(x_test_tokens, maxlen = max_tokens, padding = pad, truncating = pad)
print(x_train_pad.shape) #shape (25000, 544), 25000 word seq each containing up to 10000 most popular words, which are tokenized to int-form, and each has a len 537
print(x_train_pad[1])

#Require inverse mapping from integer tokens back to words to reconstruct text-string
idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys())) #create tuple of int to word 
def tokens_to_string(tokens): #should remove all tags, grammar, punc when rec str
    words = [inverse_map[token] for token in tokens if token != 0] #map token (int) back to word if > 0
    text = " ".join(words) #concatenate all
    return text 

#Build RNN
#First layer is embedding-layer used to create word vec via an encoding scheme, similar to dec to bin, RNN cannot cover all 10000 words so compress it
#Vals gen fall b/w -1.0 to 1.0
model = Sequential()
embedding_size = 8 #take in dim as all pop words (all possibile encodings), output vec, length of each tokenized sequence capped to 544
model.add(Embedding(input_dim = num_words_parse, output_dim = embedding_size, input_length = max_tokens, name = 'layer_embedding')) #
#Add GRU (gated recurrent unit), type of system that acts as floating-point mem for weights, taking last time seq input and current input (word)
#GRU form of modified LSTM (long short term mem) used for finding patterns in larger seq, or via units further back, designed to pred next seq output based on weights that modify effects of prev seq input, current input and current output
#Typical RNNs contain chains of NN for seq, however each one only dep on last
#LSTM use hor cell state line running through ALL units, must create a more complex network to modify effects of prev inp on curr
#Addl each time step req a copy of network as its weights are equal for ALL units, thus when solving grad via backprop, even small changes can cause large grad resulting in exploding/vanishing, factors into trickier design
model.add(GRU(units = 16, return_sequences = True)) #next layer also GRU, must return seq to connect 
model.add(GRU(units = 8, return_sequences = True))
model.add(GRU(units = 4))
model.add(Dense(1, activation = 'sigmoid')) #Add dense layer for val b/w 0.0 - 1.0
optimizer = Adam(lr = 1e-3)
model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy']) #bin cross entropy sets up bin classification prob for each input 
model.summary()
y_train = np.array(y_train) #x_train_pad np arr, ensure both match 
y_test = np.array(y_test)
model.fit(x_train_pad, y_train, validation_split = 0.05, epochs = 3, batch_size = 128)
result = model.evaluate(x_test_pad, y_test)
print("Accuracy: {0:.2%}".format(result[1]))

#Show misclassified text
y_pred = model.predict(x = x_test_pad[0:1000]) #first 1000 preds
y_pred = y_pred.T[0] #transpose first elem, y_pred list of n rows, combine all into n cols 
cls_pred = np.array([1.0 if p > 0.5 else 0.0 for p in y_pred]) #ensure pred can be classified as either 1.0 or 0.0
cls_true = np.array(y_test[0:1000]) #collect true vals
incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]
print(len(incorrect))
idx = incorrect[0]
print(idx) #print ind of first incorrectly classified text seq
text = x_test_text[idx] #print seq 
print(text)
print(y_pred[idx])
print(cls_true[idx])

#Testing with new data
text1 = "This movie is fantastic! I really like it because it is so good!"
text2 = "Good movie!"
text3 = "Maybe I like this movie."
text4 = "Meh ..."
text5 = "If I were a drunk teenager then this movie might be good."
text6 = "Bad movie!"
text7 = "Not a good movie!"
text8 = "This movie really sucks! Can I get my money back please?"
texts = [text1, text2, text3, text4, text5, text6, text7, text8]

tokens = tokenizer.texts_to_sequences(texts)
tokens_pad = pad_sequences(tokens, maxlen = max_tokens, padding = pad, truncating = pad) #shape will be (8, 537), each text seq w upto 537 tokens for its encoded words
print(model.predict(tokens_pad))



