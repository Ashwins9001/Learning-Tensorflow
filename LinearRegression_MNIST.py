import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data

"Convert to one-hot encoding to easily pick out number via index"
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
print("Training Set: {} ".format(len(data.train.labels)))
print("Test Set: {} ".format(len(data.test.labels)))
print("Validation Set: {} ".format(len(data.validation.labels)))

"Return index of maximum val, one hot therefore return num itself per each ROW"
"Array of classes for resp images"
data.test.cls = np.array([label.argmax() for label in data.test.labels])

"Defn MNIST img size, flatten to 1D, create tuple, defn possible nums (classes)"
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10

"Helper func to plot img; nine in a 3x3 grid"
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    "Enumerate to track obj and index, flatten rows, cols to loop over all nine"
    for i, ax in enumerate(axes.flat):
        "Reshape to retrieve img dims"
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
        
images_data = data.test.images[0:9]
"Set true classes equal to this, taken from test set, for no prediction just show"
cls_true_data = data.test.cls[0:9]
plot_images(images_data, cls_true_data)
        
"Defn placeholder var that are mapped as input img to comp graph"
"A tensor (multidim vec/matrix) with float data-type 32 bits"
"None defn arbitrary rows with cols img_size_flat, thus each row holds img"
x = tf.placeholder(tf.float32, [None, img_size_flat])
"One-hot encoding for true class"
y_true = tf.placeholder(tf.float32, [None, num_classes])
"True number class"
y_true_cls = tf.placeholder(tf.int64, [None])

"Defn Var optimization"
"Weights defn linear func steepness, for small changes of that var, large effect on output"
"Class prediction per pixel"
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
"Biases defn pre-existing cond that affect output"
"Possible test set is predisposed to guessing a small set of nums"
biases = tf.Variable(tf.zeros([num_classes]))

"Defn model which returns [num_images, num_classes] given x = [num_images, img_size_flat] and weights = [img_size_flat, num_classes] thus eliminated by matrix multiplication"
"Class predictions per each img"
"ith row, jth col reps likelihood of jth class matching ith img"
"Neurons of network"
logits = tf.matmul(x, weights) + biases

"Regularization for Comparison and Preparation for Cost Min"
"Activation Function Residing in Neuron"
"Implement softmax to linearize results to a range of vals b/w [0,1], row sum=1"
y_pred = tf.nn.softmax(logits)
"Return index per each row for arr of predictions"
y_pred_cls = tf.argmax(y_pred, dimension=1)

"Cost Func Optimization"
"Compare model performance by checking y_pred against y_true"
"Cross entropy func used for this: continuous and pos, only goes to zero when predicted output == true output"
"Goal is to modify weights/biases to get cross-entropy very close to zero"
"Built-in func calculates softmax too, must use logits against y_true"
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
"Mean cross entropy of ALL input img for scalar to work to min"
cost = tf.reduce_mean(cross_entropy)
"used for weight/bias modification via backprop"
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

"Performance Measures"
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
"Cast bool arr to floating-point, FALSE=0, TRUE=1, take mean to determine ratio of correctly classified img"
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"Start TF"
session = tf.session()
session.run(tf.initialize_all_variables())

"Training Model"
"Defn batch_size to only go over a few images, each iteration will progressively modify weights/biases"
batch_size = 100
def optimize(num_iterations):
    for i in range(num_iterations):
        "x_batch holds img and y_true_batch true labels"
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        "Keys dict match placeholder var defn above"
        feed_dict_train = {x:x_batch,
                           y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        
"Helper Func for Performance Display"
feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}

def print_accuracy():
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test set: {0:.1%}".format(acc))
    
def plot_example_errors():
    "Retrieve list of bool values from corr_pred, and run on test set for error"
    "incorrect arr of bools, with each index corresponding to img"
    correct, cls_pred = session.run([correct_prediction, y_pred_cls], feed_dict=feed_dict_test)
    incorrect = (correct == False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    "Return true img from sample"
    cls_true = data.test.cls[incorrect]
    "Plot sample of first nine"
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])
print_accuracy()
plot_example_errors()
    