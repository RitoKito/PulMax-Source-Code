import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import BatchNormalization

import glob as gb

from PIL import Image

import io
import os

import datetime as datetime

import sklearn as sklearn
import sklearn.metrics

import itertools
import tensorboard as tboard
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Error(Exception):
    pass

class InvalidModel(Error):
    pass

if __name__ == "__main__":

    img_height, img_width = 224, 224
    batch_size = 16

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        './TRAIN FOLDER/',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        './VAL FOLDER/',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        './TEST FOLDER/',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False)

    while(True):
        user_input = input("\nEnter \"train\" to train a new model.\n"
                +"Enter \"evaluate\" to use a pre-existing model.\n"
                +"Enter \"x\" to exit.\n"
                +"Input: ")

        if(user_input.lower() == "train"):

            datagen = ImageDataGenerator(
                rotation_range=40,
                rescale=1./255,
                horizontal_flip=True,
                fill_mode='nearest')

            #images = []
            #for filename in gb.glob("./MiniSampleNEW/TUBERCULOSIS/*.png"):
            #    image = load_img(filename)
        #        if filename.endswith(".jpeg"):
        #            name = 'COVID19(' + str(c) + ').jpg'
        #            rgb_im = image.convert('RGB')
        #            image = image.save(name)

            #    image = load_img(filename)
            #    x = img_to_array(image)
            #    x = x.reshape((1,) + x.shape)
            #    images.append(x)


            #img = load_img('./train/COVID19/COVID19(0).jpg')
            #x = img_to_array(img)
            #x = x.reshape((1,) + x.shape)

            #i = 0

            #for item in images:
        #        for batch in datagen.flow(item, batch_size=1,
        #                                    save_to_dir='./MegaSampleNEW/TUBERCULOSIS', save_prefix='TUBERCULOSIS', save_format='jpeg'):
        #            i += 1
        #            if i > 30:
        #                i = 0
        #                break


            num_classes = 4

            data_augmentation = tf.keras.Sequential([
                layers.experimental.preprocessing.RandomRotation(0.1),
                layers.experimental.preprocessing.RandomZoom(0.05),
            ])

            model = tf.keras.Sequential([
                data_augmentation,
                tf.keras.layers.experimental.preprocessing.Rescaling(1./255),

                tf.keras.layers.Conv2D(8, (5,5), padding='same', activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.11)),
                tf.keras.layers.MaxPooling2D((2, 2), strides=2),

                tf.keras.layers.Conv2D(16, (5,5), padding='same', activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.11)),
                tf.keras.layers.MaxPooling2D((2, 2), strides=2),

                tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.11)),
                tf.keras.layers.MaxPooling2D((2, 2), strides=2),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.65),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(num_classes, activation='softmax')])

            model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

            def plot_confusion_matrix(cm, class_names):
                """
                Returns a matplotlib figure containing the plotted confusion matrix.

                Args:
                   cm (array, shape = [n, n]): a confusion matrix of integer classes
                   class_names (array, shape = [n]): String names of the integer classes
                """

                figure = plt.figure(figsize=(8, 8))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title("Confusion matrix")
                plt.colorbar()
                tick_marks = np.arange(len(class_names))
                plt.xticks(tick_marks, class_names, rotation=45)
                plt.yticks(tick_marks, class_names)

                # Normalize the confusion matrix.
                cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

                # Use white text if squares are dark; otherwise black.
                threshold = cm.max() / 2.

                for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    color = "white" if cm[i, j] > threshold else "black"
                    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                return figure

            def plot_to_image(figure):
                """
                Converts the matplotlib plot specified by 'figure' to a PNG image and
                returns it. The supplied figure is closed and inaccessible after this call.
                """

                buf = io.BytesIO()

                # Use plt.savefig to save the plot to a PNG in memory.
                plt.savefig(buf, format='png')

                # Closing the figure prevents it from being displayed directly inside
                # the notebook.
                plt.close(figure)
                buf.seek(0)

                # Use tf.image.decode_png to convert the PNG buffer
                # to a TF image. Make sure you use 4 channels.
                image = tf.image.decode_png(buf.getvalue(), channels=4)

                # Use tf.expand_dims to add the batch dimension
                image = tf.expand_dims(image, 0)

                return image

            def log_confusion_matrix(epoch, logs):

                # Use the model to predict the values from the test_images.
                test_pred_raw = model.predict(test_ds)

                test_pred = np.argmax(test_pred_raw, axis=1)

                test_labels =  np.concatenate([y for x, y in test_ds], axis=0)
                # Calculate the confusion matrix using sklearn.metrics
                cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)

                class_names = test_ds.class_names
                figure = plot_confusion_matrix(cm, class_names=class_names)
                cm_image = plot_to_image(figure)

                file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
                # Log the confusion matrix as an image summary.
                with file_writer_cm.as_default():
                    tf.summary.image("Confusion Matrix", cm_image, step=epoch)

            tensor_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=tensor_callback)
            #history = model.fit(train_ds, validation_data=val_ds, epochs=10)

            model.save('CNNModel '+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

            print('Evaluating.')
            model.evaluate(test_ds)


            while(True):
                specs = input('Would you like to see the model\'s summary?\n'
                +'Enter \"y\" for yes.\n'
                +'Enter \"n\" for no.\n'
                +'Input: ')
                if(specs.lower() == 'y'):
                    model.summary()
                    tf.keras.backend.clear_session()
                    break
                elif(specs.lower() == 'n'):
                    tf.keras.backend.clear_session()
                    print('\n')
                    break
                else:
                    print('Incorrect Input.')
                    continue

        elif(user_input.lower() == 'evaluate'):
            while(True):
                try:
                    model_path = input('Enter path to a model: ')
                    if(os.path.exists('./' + model_path)):
                        pass
                    else:
                        raise InvalidModel

                    model = models.load_model('./'+model_path)
                    print('\nModel successfuly loaded.')
                    print('Evaluating.\n')
                    model.evaluate(test_ds)

                    specs = input('Would you like to see model\'s summary?\n'
                    +'Enter \"y\" for yes.\n'
                    +'Enter \"n\" for no.\n'
                    +'Input: ')
                    while(True):
                        if(specs.lower() == 'y'):
                            model.summary()
                            break
                        elif(specs.lower() == 'n'):
                            print('\n')
                            break
                        else:
                            print('Incorrect Input')
                            continue
                    break
                except Exception as e:
                    print(e)
                    continue
        elif(user_input.lower() == 'x'):
            break
        else:
            print('Incorrect Input.')
            continue
