import tensorflow as tf
import os
from dipy.io.streamline import load_tractogram, save_tractogram
import matplotlib.pyplot as plt
import src.utils.constants as constants
from src.models.generators import SLGenerator

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.histo3.ry['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

# Small recurrent model
class lstmClassifier(tf.keras.Model):
    _emb_size= 32
    _rnn_size= 32
    _int_size= 32
    _s_batch = 12

    def __init__(self):
        super(lstmClassifier, self).__init__(name="lstm_classifier")
        nclasses      = constants.clusters['size']+1
        self.embedding= tf.keras.layers.Dense(self._emb_size)
        self.lstm     = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self._rnn_size))
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.dense    = tf.keras.layers.Dense(self._int_size, activation='relu')
        self.final    = tf.keras.layers.Dense(nclasses)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout1(x)
        x = self.lstm(x)
        x = self.dropout2(x)
        x = self.dense(x)
        x = self.final(x)
        return x

    def train(self, train_subjects, test_subjects, retrain=True):
        self.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=tf.keras.optimizers.Adam(1e-4),
                     metrics=['accuracy'])
        ### Checkpoints
        checkpoint_dir   = constants.models_path+'checkpoints-LSTM'
        checkpoint_prefix= os.path.join(checkpoint_dir,"ckpt")
        checkpoint       = tf.train.Checkpoint(optimizer=self.optimizer,model=self)

        if retrain==True:
            print('[INFO] Used for training:', train_subjects)
            print('[INFO] Used for testing:', test_subjects)

            # Training
            training_generator = SLGenerator(train_subjects)
            validation_generator = SLGenerator(test_subjects, batchsize=10000)
            history = self.fit(training_generator, epochs=25, validation_data=validation_generator)
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
            # Load parameters saved in the previous session
            print("[INFO] Restoring last model")
            status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

if __name__ == '__main__':
    print('Training')
    lstm = lstmClassifier()
    train = ['151425/']
    test = ['155938/']
    lstm.train(train, test)