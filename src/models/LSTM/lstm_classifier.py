import tensorflow as tf
import os
from dipy.io.streamline import load_tractogram, save_tractogram
import matplotlib.pyplot as plt

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

    def __init__(self,classes):
        super(lstmClassifier, self).__init__(name="lstm_classifier")
        nclasses      = len(classes)
        self.embedding= tf.keras.layers.Dense(self.emb_size)
        self.lstm     = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size))
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.dense    = tf.keras.layers.Dense(self.int_size, activation='relu')
        self.final    = tf.keras.layers.Dense(nclasses)
        self.classes  = classes

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout1(x)
        x = self.lstm(x)
        x = self.dropout2(x)
        x = self.dense(x)
        x = self.final(x)
        return x

    def _get_train_dataset(self, subjects):
        for subject in subjects:
            print('[INFO] Reading subject:',subject)                
            ### Reads the .npy files from each specified class
            for i,c in enumerate(self.classes):
                ### Load tractogram
                #filename   = path_files+'auto'+c+'.tck'
                filename   = path_files+subject+'/'+c+'_20p.tck'
                if not os.path.isfile(filename):
                    continue
                print('[INFO] Reading file:',filename)
                #tractogram = load_tractogram(filename, path_files+fNameRef, bbox_valid_check=False)
                tractogram = load_tractogram(filename, './utils/t1.nii.gz', bbox_valid_check=False)
                # Get all the streamlines
                STs      = tractogram.streamlines
                if subject==test_subject:
                    val_trajs.extend(scaledSTs)
                    val_labels.extend(len(scaledSTs)*[i])
                else:
                    train_trajs.extend(scaledSTs)
                    train_labels.extend(len(scaledSTs)*[i])

        train_dataset = tf.data.Dataset.from_tensor_slices((train_trajs,train_labels))
        train_dataset       = train_dataset.shuffle(600000, reshuffle_each_iteration=False)
        train_dataset       = train_dataset.batch(s_batch)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_trajs,val_labels))
        val_dataset       = val_dataset.batch(s_batch)

    def train(self,train_subjects,test_subject,retrain=True):
        self.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     optimizer=tf.keras.optimizers.Adam(1e-4),
                     metrics=['accuracy'])
        ### Checkpoints
        checkpoint_dir   = './checkpoints-LSTM'
        checkpoint_prefix= os.path.join(checkpoint_dir,"ckpt")
        checkpoint       = tf.train.Checkpoint(optimizer=self.optimizer,model=self)

        if retrain==True:
            train_dataset = self._get_train_dataset(train_subjects)
            val_dataset = self._get_val_dataset(test_subject)
            print('[INFO] Used for testing:', test_subject)
            print('[INFO] Total number of streamlines for training:', train_dataset.shape[0])
            print('[INFO] Total number of streamlines for validation:', val_dataset.shape[0])

            # Training
            history = self.fit(train_dataset, epochs=25, validation_data=val_dataset)
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