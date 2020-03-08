import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import PIL
import keras
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
import knifey
from sklearn.utils.class_weight import compute_class_weight


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
    images = [plt.imread(path) for path in image_paths]
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
train_dir = knifey.train_dir #load set amt img locally from another set for model to transfer & pred
test_dir = knifey.test_dir

#Set up VGG-16 full-model, if include_top = False include only conv layers, FCN excluded
model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet') #require input tensor = 224 x 224 x 3

input_shape = model.layers[0].output_shape[1:3] #verify, ret first layer (in) and check rem two params for size
print(input_shape)
datagen_train = ImageDataGenerator( #Create data generator for augmented set, defn img transform
    rescale = 1./255,
    rotation_range = 180,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = [0.9, 0.5],
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = 'nearest')
datagen_test = ImageDataGenerator(rescale = 1./255) #Don't want aug test img, need acc for pred, ensure scaled to range 0.0 - 1.0 as expected by VGG-16
batch_size = 20 #load in batches to not overgo RAM
if True:
    save_to_dir = None
else:
    save_to_dir = 'augmented_images/'
generator_train = datagen_train.flow_from_directory(directory = train_dir, #instantiate gen w img
                                                    target_size = input_shape,
                                                    batch_size = batch_size,
                                                    shuffle = False)
print(generator_train) #4170 img belong to 3 classes 
generator_test = datagen_train.flow_from_directory(directory = test_dir, #instantiate gen w img
                                                    target_size = input_shape,
                                                    batch_size = batch_size,
                                                    shuffle = False)
print(generator_test) #530 img belong to 3 classes
steps_for_test = generator_test.n / batch_size
image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)
cls_train = generator_train.classes #class num for all img in set
cls_test = generator_test.classes
class_names = list(generator_train.class_indices.keys()) #labels assoc w class num, store in list
num_classes = generator_train.num_classes
images = load_images(image_paths = image_paths_train[0:9]) #load nine img
cls_true = cls_train[0:9]
plot_images(images = images, cls_true = cls_true, smooth = True)

#Knifey dataset containts more img of spoons & forks than knives, model biased to id one class
#Calc weights balance dataset, applied to each img grad to have influence on overall grad
class_weight = compute_class_weight(class_weight = 'balanced', classes = np.unique(cls_train), y = cls_train) #apply np.unique() as to not repeat classes, apply to all train img
print(class_weight) #[1.39, 1.15, 0.71] for [forkey, knify, spoony]
print(class_names)

#Helper func to load, resize img for VGG-16 compatibility and completing prediction
def predict(image_path):
    img = PIL.Image.open(image_path)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)
    plt.imshow(img_resized)
    plt.show()
    img_arr = np.expand_dims(np.array(img_resized), axis = 0) #plot as np arr, expand dim w new axis
    pred = model.predict(img_arr) #using VGG-16 outputs class breakdown out of 100% as per 1000 classes, all vals inside arr
    pred_decoded = decode_predictions(pred)[0] #defn by keras, decode preds provides class name of highest match (first elem); multiple img can exist, thus index to zero, only one for now
    for code, name, score in pred_decoded:
        print("{0:>6.2%} : {1}".format(score, name))
predict(image_path = 'data/parrot.jpg')
predict(image_path = image_paths_train[0])
predict(image_path = image_paths_train[1])
predict(image_path = image_paths_test[0])
model.summary() #check struc of VGG16 NN to get ref to last conv layer by name
#shape matches that of block5_pool: (7, 7, 512)

conv_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet') #require input tensor = 224 x 224 x 3


fc = Dense(1024, activation = 'relu')(conv_model)
dropout = Dropout(0.5)(fc)
fc2 = Dense(num_classes, activation = 'softmax')(dropout)
new_model = Model(inputs = conv_model.input, outputs = fc2)
#conv_model.add(Flatten())
#conv_model.add(Dense(1024, activation = 'relu'))
optimizer = Adam(lr = 1e-5)
loss = 'categorical_crossentropy' #outputs probability over C classes, used for multi-class classification
metrics = ['categorical_accuracy']

def print_layer_trainable():
    for layer in conv_model.layers: #checking struc of conv layers in model
        print("{0}:\t{1}".format(layer.trainable, layer.name)) #check if layer is trainable or has been frozen (weights cannot be updated)
print_layer_trainable()