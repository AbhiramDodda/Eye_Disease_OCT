from glob import glob
import pandas as pd
import numpy as np
from numpy import expand_dims
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage.color import gray2rgb
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from keras import applications, optimizers
from keras.models import Model, Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Dense, Flatten, Dropout
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import to_categorical, model_to_dot, plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

data_dir = "../input/kermany2018/OCT2017 /"
train_path= '../input/kermany2018/OCT2017 /train/'
val_path= '../input/kermany2018/OCT2017 /val/'
test_path= '../input/kermany2018/OCT2017 /test/'
img_width, img_height = 150, 150 
channels = 3
batch_size = 32


# Visualiizing Data distribution across 4 classes
cnv_images = len(glob(train_path + 'CNV/*.jpeg'))
dme_images = len(glob(train_path + 'DME/*.jpeg'))
drusen_images = len(glob(train_path + 'DRUSEN/*.jpeg'))
normal_images = len(glob(train_path + 'NORMAL/*.jpeg'))
data= {'CNV': cnv_images, 'DME': dme_images, 'DRUSEN': drusen_images, 'NORMAL': normal_images}
labels = list(data.keys()) 
count = list(data.values()) 

plt.rcParams['figure.figsize'] = (8.0, 8.0)
plt.bar(labels, count, color=['tab:red', 'tab:green', 'tab:blue', 'tab:orange'])
plt.axis('on')
plt.xlabel("Labels") 
plt.ylabel("Count") 
plt.savefig('labels_vs_counts.png', transparent= False, bbox_inches= 'tight', dpi= 400)
plt.show() 



train = ImageDataGenerator(horizontal_flip = True, rescale = 1/255, fill_mode = 'nearest')
train_data = train.flow_from_directory(train_path, target_size = (img_width, img_height), batch_size = 256, color_mode='rgb', class_mode="categorical", shuffle = True)
test = ImageDataGenerator(rescale = 1/255, fill_mode = 'nearest')
test_data = test.flow_from_directory(test_path, target_size = (img_width, img_height), batch_size = 256, color_mode='rgb', class_mode="categorical", shuffle = True)
validation = ImageDataGenerator(rescale = 1/255, fill_mode = 'nearest')
validation_data = test.flow_from_directory(val_path, target_size = (img_width, img_height), batch_size = 256, color_mode='rgb', class_mode="categorical", shuffle = True)

vgg16 = VGG16(include_top= False, input_shape= (img_width, img_height, channels), weights= 'imagenet') # Pretrained on ImageNet dataset

model = Sequential()

for layer in vgg16.layers:
    model.add(layer)

for layer in model.layers:
    layer.trainable= False

model.add(Flatten(input_shape= (4, 4, 512)))
model.add(Dropout(0.2))
model.add(Dense(4,activation='softmax'))

model.summary()

model.trainable = True
model.compile(optimizer= keras.optimizers.Adam(lr= 1e-5), loss= 'categorical_crossentropy', metrics= ['accuracy'])

''' Callback to save the Keras model or model weights at some frequency. '''
checkpoint = ModelCheckpoint(
    'finetuned_model.h5',
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='auto',
    save_weights_only=False,
    period=1
)

''' Stop training when a monitored metric has stopped improving. '''
earlystop = EarlyStopping(
    monitor='loss',
    min_delta=0.001,
    patience=3,
    verbose=1,
    mode='auto'
)

''' Reduce learning rate when a metric has stopped improving. '''
reduceLR = ReduceLROnPlateau(
    monitor='loss',
    factor=0.1,
    patience=3,
    verbose=1, 
    mode='auto'
)

callbacks = [checkpoint, earlystop, reduceLR]

model_fit = model.fit(
    train_data, 
    epochs = 10,
    steps_per_epoch = 50,
    validation_data = validation_data, 
    validation_steps = 25,
    verbose = 2,
    callbacks = callbacks,
    shuffle = True
)

# Accuracy - 95% loss - 0.0809