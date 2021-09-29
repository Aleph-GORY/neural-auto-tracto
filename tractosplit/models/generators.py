import numpy as np
from numpy.lib.npyio import load as npload
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random
import src.utils.constants as constants

class SLGenerator(tf.keras.utils.Sequence) :
  
    def __init__(self, subjects, batchsize=128, shuffle=True) :
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.subjects = list(subjects)

        if self.shuffle:
            random.shuffle(self.subjects)

        self.lenghts = []
        for subject in subjects:
            features_path = constants.data_proc_path+subject
            with open(features_path+'garbage.npy', 'rb') as f:
                x_garbage = npload(f)
                self.lenghts.append(int(np.floor(x_garbage.shape[0]*constants.garbage_percent)))
            with open(features_path+'labeled.npy', 'rb') as f:
                x_labeled = npload(f)
                self.lenghts[-1] += x_labeled.shape[0]
        self.batchs = [0]
        for len in self.lenghts:
            self.batchs.append(int(np.floor(len / self.batchsize)) + self.batchs[-1])
        self.subject_id = 0
      
    def __len__(self) :
        return self.batchs[-1]

    def _get_tracto_dataset(self, subject):
        features_path = constants.data_proc_path+subject
        
        x_garbage = None
        with open(features_path+'garbage.npy', 'rb') as f:
            x_garbage = npload(f)
        tf.random.shuffle(x_garbage)
        garbage_size = int(np.floor(x_garbage.shape[0]*constants.garbage_percent))
        x_garbage = x_garbage[0:garbage_size]
        y_garbage = np.zeros(x_garbage.shape[0],dtype=np.int32)
        x_labeled = y_labeled = None
        with open(features_path+'labeled.npy', 'rb') as f:
            x_labeled = npload(f)
            y_labeled = npload(f)

        x = np.concatenate([x_garbage, x_labeled], axis=0)
        y = np.concatenate([y_garbage, y_labeled], axis=0)
        random_indices = tf.random.shuffle(range(x.shape[0]))
        x = x[random_indices]
        y = to_categorical(y[random_indices])

        return x,y

    def __getitem__(self, idx) :
        if idx >= self.batchs[self.subject_id+1]:
            self.subject_id += 1
        idx += -self.batchs[self.subject_id]
        if idx == 0:
            self.x, self.y = self._get_tracto_dataset(self.subjects[self.subject_id])

        batch_x = self.x[idx*self.batchsize:(idx+1)*self.batchsize]
        batch_y = self.y[idx*self.batchsize:(idx+1)*self.batchsize]

        return batch_x, batch_y

    def on_epoch_end(self):
        self.subject_id = 0
        if self.shuffle:
            random.shuffle(self.subjects)

if __name__=='__main__':
    gen = SLGenerator(['151425/'])
    x, y = gen.__getitem__(0)
    print(x.shape)
    print(y.shape)