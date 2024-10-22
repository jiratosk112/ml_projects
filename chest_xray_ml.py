#*****************************************************************************
# Filename: chest_xray_ml.py
# Author: Jirat Boomuang
# Email: jirat_boomuang@sloan.mit.edu
# Description: For training and testing chest x-ray ML
#*****************************************************************************

#-- Import libraries ---------------------------------------------------------
import os, shutil
import random
import numpy as np
import pandas as pd
import cv2
import skimage
import matplotlib.pyplot as plt
import skimage.segmentation
import seaborn as sns

#-- tensorflow libraries --
print(f"\n\n-- importing tensorflow --")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D,Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
print(f"\n-- tensorflow imported --")
#-----------------------------------------------------------------------------

#-- Define constants ---------------------------------------------------------
labels = ['PNEUMONIA', 'NORMAL']
image_size = 128
#-----------------------------------------------------------------------------

#-- Define functions ---------------------------------------------------------

#-- define get_data() --
def get_data(data_dir):
    data=[]
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for image in os.listdir(path):
            try:
                #print(f"path = {os.path.join(path, image)}")
                image_arr = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                if image_arr is None:
                    print(f"*** Cannot read image_arr *** path = {os.path.join(path, image)}")
                    continue
                else: 
                    resized_arr = cv2.resize(image_arr, (image_size, image_size))
                    data.append([resized_arr, class_num])
                #-- End of if-else --
            except Exception as e:
                print(e)
    return np.array(data, dtype="object")
#-- End of get_data() --

#----------------------------------------------------------------------------- 

if __name__ == "__main__":
    # #%matplotlib inline
    # plt.style.use('ggplot')
    # print(f"\n\n---- Hello World ----\n\n")

    #-- Retrieve data from all folders --
    print(f"\nRetrieve data from all three folders")
    train = get_data("chest_xray\\chest_xray\\train")
    test = get_data("chest_xray\\chest_xray\\test")
    val = get_data("chest_xray\\chest_xray\\val")
    print(f"\n[DONE] Retrieve data from all three folders")

    #-- Generate batches of tensor image for data augmentation --
    print(f"\nGenerate batches of tensor image")
    train_datagen = ImageDataGenerator(rescale = 1. / 255, 
                  horizontal_flip=0.4,
                  vertical_flip=0.4,
                  rotation_range=40,
                  shear_range=0.2,
                  width_shift_range=0.4,
                  height_shift_range=0.4,
                  fill_mode="nearest")
    valid_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    #-- Generate tensor images directly from the specified folders --
    train_generator = train_datagen.flow_from_directory("chest_xray/chest_xray/train",
                                 batch_size = 32,
                                 target_size=(128,128),
                                 class_mode = 'categorical',
                                 shuffle=True,
                                 seed = 42,
                                 color_mode = 'rgb')
    valid_generator = valid_datagen.flow_from_directory("chest_xray/chest_xray/val",
                                 batch_size = 32,
                                 target_size=(128,128),
                                 class_mode = 'categorical',
                                 shuffle=True,
                                 seed = 42,
                                 color_mode = 'rgb')
    print(f"\n[DONE] Generate batches of tensor image")

    #-- Assign class labels --
    class_labels = train_generator.class_indices
    class_name = {value:key for (key, value) in class_labels.items()}

    #-- Define VGG19 Architecture for Deep CNN --
    base_model = VGG19(input_shape = (image_size,image_size,3),
                     include_top = False,
                     weights = 'imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    flat = Flatten()(x) #-- flatten vectors --


    class_1 = Dense(4608, activation = 'relu')(flat)
    dropout = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation = 'relu')(dropout)
    output = Dense(2, activation = 'softmax')(class_2)

    model_01 = Model(base_model.inputs, output)
    model_01.summary()

    #-- Start training --
    filepath = "model.weights.h5.keras"

    try:
        es = EarlyStopping(monitor="val_loss", verbose=1, mode="min", patience=4)
        
        cp=ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True, save_weights_only=False ,mode="auto", save_freq="epoch")
        lrr = ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=0.0001)
        
        sgd = SGD(learning_rate=0.0001, decay = 1e-6, momentum=0.00001, nesterov = True)
        
        model_01.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])
        
    except Exception as e:
        print(e)

    history_01 = model_01.fit(train_generator, 
            steps_per_epoch=50,
            epochs=1, 
            callbacks=[es, cp, lrr],
            validation_data=valid_generator)
    
    if not os.path.isdir('model_weights/'):
        os.mkdir("model_weights/")

    model_01.save(filepath = "model_weights/vgg19_model_01.h5.keras", overwrite=True)

    test_generator = test_datagen.flow_from_directory("chest_xray/chest_xray/test",
                                 batch_size = 32,
                                 target_size=(128,128),
                                 class_mode = 'categorical',
                                 shuffle=True,
                                 seed = 42,
                                 color_mode = 'rgb')
    
    model_01.load_weights("model_weights/vgg19_model_01.h5.keras")

    vgg_val_eval_01 = model_01.evaluate(valid_generator)
    vgg_test_eval_01 = model_01.evaluate(test_generator)

    #-- Print results --
    print(f"\n")
    print("*****************************************************************************")
    print("Display Initial Results")
    print("*****************************************************************************")
    print(f"Validation Loss: {vgg_val_eval_01[0]}")
    print(f"Validation Accuarcy: {vgg_val_eval_01[1]}")
    print(f"Test Loss: {vgg_test_eval_01[0]}")
    print(f"Test Accuarcy: {vgg_test_eval_01[1]}")
    print("=============================================================================")
    print(f"\n")

    #-- Fine Tuning --
    # Increamental unfreezing & fine tuning
    base_model = VGG19(include_top=False, input_shape=(128,128,3))
    base_model_layer_names = [layer.name for layer in base_model.layers]

    x = base_model.output
    flat = Flatten()(x)


    class_1 = Dense(4608, activation = 'relu')(flat)
    dropout = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation = 'relu')(dropout)
    output = Dense(2, activation = 'softmax')(class_2)

    model_02 = Model(base_model.inputs, output)
    model_02.load_weights("model_weights/vgg19_model_01.h5.keras")

    set_trainable = False
    for layer in base_model.layers:
        if layer.name in [ 'block5_conv3','block5_conv4']:
            set_trainable=True
        if set_trainable:
            set_trainable=True
        else:
            set_trainable=False
    print(model_02.summary())


    sgd = SGD(learning_rate=0.0001, decay = 1e-6, momentum=0.000001, nesterov = True)

    model_02.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

    history_02 = model_02.fit(train_generator, 
            steps_per_epoch=10,
            epochs=1, 
            callbacks=[es, cp, lrr],
            validation_data=valid_generator)
    
    if not os.path.isdir('model_weights/'):
        os.mkdir("model_weights/")
    model_02.save(filepath = "model_weights/vgg19_model_02.h5.keras", overwrite=True)


    model_02.load_weights("model_weights/vgg19_model_02.h5.keras")

    vgg_val_eval_02 = model_02.evaluate(valid_generator)
    vgg_test_eval_02 = model_02.evaluate(test_generator)

    print(f"\n")
    print("*****************************************************************************")
    print("Fine Tuning Results")
    print("*****************************************************************************")
    print(f"Validation Loss: {vgg_val_eval_02[0]}")
    print(f"Validation Accuarcy: {vgg_val_eval_02[1]}")
    print(f"Test Loss: {vgg_test_eval_02[0]}")
    print(f"Test Accuarcy: {vgg_test_eval_02[1]}")
    print("=============================================================================")
    print(f"\n")


    # Unfreezing and fine tuning the entire network
    base_model = VGG19(include_top=False, input_shape=(128,128,3))

    x = base_model.output
    flat = Flatten()(x)

    class_1 = Dense(4608, activation = 'relu')(flat)
    dropout = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation = 'relu')(dropout)
    output = Dense(2, activation = 'softmax')(class_2)

    model_03 = Model(base_model.inputs, output)
    model_03.load_weights("model_weights/vgg19_model_01.h5.keras")

    print(model_03.summary())

    sgd = SGD(learning_rate=0.0001, decay = 1e-6, momentum=0.0000001, nesterov = True)

    model_03.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

    history_03 = model_03.fit(train_generator, 
            steps_per_epoch=100,
            epochs=1, 
            callbacks=[es, cp, lrr],
            validation_data=valid_generator)
    
    if not os.path.isdir('model_weights/'):
        os.mkdir("model_weights/")
    model_03.save(filepath = "model_weights/vgg_unfrozen.h5.keras", overwrite=True)

    model_03.load_weights("model_weights/vgg19_model_02.h5.keras")

    vgg_val_eval_03 = model_03.evaluate(valid_generator)
    vgg_test_eval_03 = model_03.evaluate(test_generator)

    print(f"\n")
    print("*****************************************************************************")
    print("Last Round Results")
    print("*****************************************************************************")
    print(f"Validation Loss: {vgg_val_eval_02[0]}")
    print(f"Validation Accuarcy: {vgg_val_eval_02[1]}")
    print(f"Test Loss: {vgg_test_eval_02[0]}")
    print(f"Test Accuarcy: {vgg_test_eval_02[1]}")
    print("=============================================================================")
    print(f"\n")

#-- End of if __name__ -------------------------------------------------------

#*****************************************************************************
# End of File
#*****************************************************************************