# -*- coding: utf-8 -*-
"""Copy of CNN_brain_team.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nRHXuWu5gYRxLdwt-MZKcq6ch9uO9Hb2
"""

import tensorflow as tf
import sys
from dipy.io.streamline import load_tractogram
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import datasets, layers, models
from dipy.io.streamline import load_tractogram,save_tractogram
from dipy.align.streamlinear import set_number_of_points
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# Plot metrics
def Loss_ACC_plot(history):
	# plot loss
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='validation')
    # plot accuracy
    plt.subplot(1,2,2)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='validation')

def get_fibermap(all_trajs, n):

    fiber_map = np.zeros([20, 40, 3]) # define empty map for each streamline - only 10 tiles
    all_fibre_map = np.zeros([len(all_trajs), 20, 40, 3]) # to store all maps

    for j in range(len(all_trajs)):

        data = all_trajs[j] # choose one streamline

        for i in range(3): # for each dimension in streamline
            stream = data[:,i]
            stream_rev = stream[::-1] # reverse

            block1 = np.concatenate((stream, stream_rev), axis = 0) # build blocks
            block2 = np.concatenate((stream_rev, stream), axis = 0)

            cell = np.vstack((block1, block2)) # stack vertically

            fiber_slice = np.tile(cell, (10,1)) # create fiber map

            fiber_map[:,:,i] = fiber_slice # assign to map for each dimension

        all_fibre_map[j,:,:,:] = fiber_map # save all maps from all streamlines

    return all_fibre_map


# Our small recurrent model
class cnnClassifier(tf.keras.Sequential):
    def __init__(self,classes):
        super(cnnClassifier, self).__init__(name="cnnClassifier")
        self.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(20,40,3)))
        self.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.add(MaxPooling2D((2, 2)))
        self.add(Dropout(0.2))
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.add(MaxPooling2D((2, 2)))
        self.add(Dropout(0.2))
        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.add(MaxPooling2D((2, 2)))
        self.add(Dropout(0.2))
        self.add(Flatten())
        self.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.add(Dropout(0.2))
        self.add(Dense(35, activation='softmax'))
        self.classes  = classes

    def train(self,subjects,test_subject,path_files,retrain=True):
        opt = SGD(learning_rate=0.001, momentum=0.9)
        self.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # Checkpoints
        checkpoint_dir   = './checkpoints-CNN'+'-'+test_subject
        checkpoint_prefix= os.path.join(checkpoint_dir,"ckpt")
        checkpoint       = tf.train.Checkpoint(optimizer=self.optimizer,model=self)
        if retrain==True:
            train_trajs  = []
            train_labels = []
            val_trajs  = []
            val_labels = []
            for k,subject in enumerate(subjects):
                print('[INFO] Reading subject:',subject)
                # Reads the .tck files from each specified class
                for i,c in enumerate(self.classes):
                    # Load tractogram
                    #filename   = path_files+'auto'+c+'.tck'
                    filename   = path_files+subject+'/'+c+'_20p.tck'
                    if not os.path.isfile(filename):
                        continue
                    print('[INFO] Reading file:',filename)
                    #tractogram = load_tractogram(filename, path_files+fNameRef, bbox_valid_check=False)
                    tractogram = load_tractogram(filename, './utils/t1.nii.gz', bbox_valid_check=False)
                    # Get all the streamlines
                    STs      = tractogram.streamlines
                    scaledSTs= set_number_of_points(STs,20)
                    if subject==test_subject:
                        val_trajs.extend(scaledSTs)
                        val_labels.extend(len(scaledSTs)*[i])
                    else:
                        train_trajs.extend(scaledSTs)
                        train_labels.extend(len(scaledSTs)*[i])
            streamlines = get_fibermap(train_trajs, 10)
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split (streamlines, np.array(all_labels) , test_size=0.2, train_size=0.8 )

            print('[INFO] Used for testing: ',test_subject)
            print('[INFO] Total number of streamlines for training:',len(train_trajs))
            print('[INFO] Total number of streamlines for validation:',len(val_trajs))
            train_dataset = tf.data.Dataset.from_tensor_slices((train_trajs,train_labels))
            train_dataset       = train_dataset.shuffle(600000, reshuffle_each_iteration=False)
            train_dataset       = train_dataset.batch(s_batch)
            val_dataset = tf.data.Dataset.from_tensor_slices((val_trajs,val_labels))
            val_dataset       = val_dataset.batch(s_batch)

            # Training
            history = self.fit(train_dataset, epochs=35, validation_data=val_dataset)
            checkpoint.save(file_prefix = checkpoint_prefix)
            # Plots
            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
            plot_graphs(history, 'accuracy')
            plt.ylim(None, 1)
            plt.subplot(1, 2, 2)
            plot_graphs(history, 'loss')
            plt.ylim(0, None)
            plt.show()
        else:
            # To avoid training, we can just load the parameters we saved in the previous session
            print("[INFO] Restoring last model")
            status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
