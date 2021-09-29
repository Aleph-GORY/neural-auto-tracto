import numpy as np
from numpy.lib.npyio import load as npload
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random
import tractosplit.utils.constants as constants


class SL_generator(tf.keras.utils.Sequence):
    def __init__(self, subjects, batchsize=256, shuffle=True):
        self.batchsize = batchsize // 2
        self.shuffle = shuffle
        self.subjects = list(subjects)

        if self.shuffle:
            random.shuffle(self.subjects)

        self.lenghts = []
        for subject in self.subjects:
            features_path = constants.data_processed_path + subject
            x_labeled = npload(features_path + "labeled.npy", mmap_mode="r")
            self.lenghts.append(x_labeled.shape[0])

        batches = [0]
        for length in self.lenghts:
            batches.append(int(np.floor(length / self.batchsize)) + batches[-1])
        self.n_batches = batches[-1]
        self.batches = [
            range(batches[i], batches[i + 1]) for i in range(len(batches) - 1)
        ]

    def __len__(self):
        return self.n_batches

    def _get_tracto_dataset(self, subject_id, index):
        features_path = constants.data_processed_path + self.subjects[subject_id]

        x_garbage = npload(features_path + "garbage.npy", mmap_mode="r")
        x_garbage = np.array(
            x_garbage[index * self.batchsize : (index + 1) * self.batchsize]
        )
        y_garbage = np.zeros(x_garbage.shape[0], dtype=np.int32)

        with open(features_path + "labeled.npy", "rb") as f:
            x_labeled = npload(f)
            y_labeled = npload(f)
        x_labeled = x_labeled[index * self.batchsize : (index + 1) * self.batchsize]
        y_labeled = y_labeled[index * self.batchsize : (index + 1) * self.batchsize]

        x = np.concatenate([x_garbage, x_labeled], axis=0)
        y = np.concatenate([y_garbage, y_labeled], axis=0)
        y = to_categorical(y, 36)
        return x, y

    def __getitem__(self, index):
        subject = 0
        while subject < len(self.subjects):
            if index in self.batches[subject]:
                index = index - self.batches[subject].start
                break
            subject += 1

        batch_x, batch_y = self._get_tracto_dataset(subject, index)
        return batch_x, batch_y

    def on_epoch_end(self):
        self.subject_id = 0
        if self.shuffle:
            random.shuffle(self.subjects)


if __name__ == "__main__":
    gen = SL_generator(["151425/"])
    x, y = gen.__getitem__(0)
    print(x.shape)
    print(y.shape)
