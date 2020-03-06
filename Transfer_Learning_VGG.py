import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import PIL
import keras
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop
import knifey
from knifey import num_classes


#Func for joining dir to list of filenames 
def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]
#Func plot 9 img and write true & pred cls below each
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)
    fig, axes = plt.subplots(3,3)
    if cls_pred is None: #Adjust spacing depending on available data
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace = hspace, wspace = 0.3)
    #Interpolation type
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'
    for i, ax in enumerate (axes.flat):
        if i < len(images): #Ensure less than 9 img
            ax.imshow(images[i], interpolation = interpolation)
            cls_true_name = class_names[cls_true[i]]
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name) #Name predicted class
            else:
                cls_pred_name = class_names[cls_pred[i]] 
                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
            ax.set_xlabel(xlabel) #show classses as x-axis label
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
#Func to print conf matrix
def print_confusion_matrix(cls_pred): #input array of pred class num for all img
    cm = confusion_matrix(y_true = cls_test, y_pred = cls_pred) #defn true & pred cls
    print("Confusion Matrix")
    print(cm)
    for i, class_name in enumerate(class_names):
        print("({0}) {1}".format(i, class_name))
#Func to plot example errors
def plot_example_errors(cls_pred):
    incorrect = (cls_pred != cls_test) #bool arr whether pred class incorrect
    wrong_image_paths = np.array(image_path_tests)[incorrect] #get file paths for incorrect img
    cls_pred = cls_pred[incorrect] #pred & true incorrects
    cls_true = cls_test[incorrect] 
    plot_images(images = images, cls_true = cls_true[0:9], cls_pred = cls_pred[0:9]) #plot_images only loads 9
#Func to defn errors in examples
def example_errors():
    #Three types Keras gen: fit_generator, evalulate_generator, predict_generator
    #fit_gen: Instance of keras seq, require two gens one for training, one for validation data; both return tuple (inputs, targets)
    #eval:gen: Same as training_gen; pred_gen: Return only inputs
    #Generators load large sets data inc to reduce mem load on cpu
    #Must reset gen before processing as it loops infinitely and keeps internal index and may begin in middle of test set; cannot match img to pred_cls
    #Gens addl increase generality by actively augmenting single img multiple times, then replacing training set with new, changed one
    generator_test.reset()
    y_pred = new_model.predict_generator(generator_test, steps = steps_test)
    cls_pred = np.argmax(y_pred, axis = 1) #select most prob cls & convert to int
    plot_example_errors(cls_pred)
    print_confusion_matrix(cls_pred)
#Func for loading img from disk
def load_images(image_paths):
    images = [plt.imread(path) for path in image_path]
    return np.asarray(images)
#Func to plot acc & loss vals over training set, test set
def plot_training_history(history): 
    acc = history.history['categorical_accuracy'] #for training set
    loss = history.history['loss']
    val_acc = history.history['val_categorical_accuracy'] #for test set
    val_loss = history.history['val_loss']
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.') #plot for training set
    plt.plot(loss, 'o', color='b', label='Training Loss')
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.') #plot for test set
    plt.plot(val_loss, 'o', color='r', label='Test Loss')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.show()
#Dl knifey dataset, takes img from frames of vids, copy files to directories
#This dataset replace FCN of VGG-16, transfer model, pred whether knifey or forky or spoony
knifey.data_dir = "data/knifey-spoony/"
knifey.maybe_download_and_extract()
knifey.copy_files()
train_dir = knifey.train_dir
test_dir = knifey.test_dir
model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')
