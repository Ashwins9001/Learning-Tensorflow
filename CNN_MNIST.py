#Conv Layer 1
filter_size1 = 5
num_filters = 16

#Conv Layer 2 
#Take sliding sum by applying all one filter from each channel (16 available)
#Repeated 36 times to cover all filters 
filter_size2 = 5
num_filters = 36

#FC (defn neurons)
fc_size = 128

from tensorflow.examples.tutorials.mnist import input_data 
data = input_data.read_data_sets('data/MNIST', one_hot=True)

print("Training Set: {}".format(len(data.train.labels)))
print("Test Set: {}".format(len(data.test.labels)))
print("Validation Set: {}".format(len(data.validation.labels)))
