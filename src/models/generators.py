import numpy as np
from numpy.lib.npyio import load as npload
import tensorflow as tf
import src.utils.constants as constants

class My_Custom_Generator(tf.keras.utils.Sequence) :
  
    def __init__(self, subjects) :
        self.subjects = subjects
      
    def __len__(self) :
        return len(self.subjects)

    def __getitem__(self, idx) :
        features_path = constants.data_proc_path+self.subjects[idx]+'/'+self.subjects[idx]
        x_garbage = None
        with open(features_path+'_garbage.npy', 'rb') as f:
            x_garbage = npload(f)
        x_labeled = y_labeled = None
        with open(features_path+'_labeled.npy', 'rb') as f:
            x_training = npload(f)
            y_training = npload(f)
            print(x_training.shape)
            print(y_training.shape)
        batch_x = batch_y = None
        return [batch_x, batch_y]

if __name__=='__main__':
  print(constants.x)