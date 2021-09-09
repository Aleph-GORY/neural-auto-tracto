import numpy as np
import tensorflow as tf
import tractosplit.utils.constants as constants

class My_Custom_Generator(tf.keras.utils.Sequence) :
  
  def __init__(self, subjects) :
    self.subjects = subjects
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return 1

if __name__=='__main__':
  print(constants.x)