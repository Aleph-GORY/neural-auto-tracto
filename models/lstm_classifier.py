import tensorflow as tf
import os
from dipy.io.streamline import load_tractogram,save_tractogram
from dipy.align.streamlinear import set_number_of_points

samples = 3
emb_size= 32
rnn_size= 32
int_size= 32
s_batch = 128

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

# Our small recurrent model
class lstmClassifier(tf.keras.Model):
    def __init__(self,classes):
        super(lstmClassifier, self).__init__(name="lstmClassifier")
        nclasses      = len(classes)
        self.embedding= tf.keras.layers.Dense(emb_size)
        self.lstm     = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_size))
        self.dense    = tf.keras.layers.Dense(int_size, activation='relu')
        self.final    = tf.keras.layers.Dense(nclasses)
        self.classes  = classes

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense(x)
        x = self.final(x)
        return x

    def train(self,subjects,excluded,path_files,reload_weights=False):
        self.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=['accuracy'])
        # Checkpoints
        checkpoint_dir   = './checkpoints'+'-'+subjects[0]
        checkpoint_prefix= os.path.join(checkpoint_dir,"ckpt")
        checkpoint       = tf.train.Checkpoint(optimizer=self.optimizer,model=self)
        train_trajs  = []
        train_labels = []
        for k,subject in enumerate(subjects):
            if k==excluded:
                continue
            # Reads the .tck files from each specified class
            for i,c in enumerate(self.classes):
                # Load tractogram
                #filename   = path_files+'auto'+c+'.tck'
                filename   = path_files+subject+'/'+c+'_20p.tck'
                if not os.path.isfile(filename):
                    continue
                print('[INFO] Reading file:',filename)
                #tractogram = load_tractogram(filename, path_files+fNameRef, bbox_valid_check=False)
                tractogram = load_tractogram(filename, './LSTM_representation_classification/connectome_test/t1.nii.gz', bbox_valid_check=False)
                # Get all the streamlines
                STs      = tractogram.streamlines
                scaledSTs= set_number_of_points(STs,20)
                train_trajs.extend(scaledSTs)
                train_labels.extend(len(scaledSTs)*[i])

        print('[INFO] Used for testing: ',subjects[excluded])
        print('[INFO] Total number of streamlines for training:',len(train_trajs))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_trajs,train_labels))
        train_dataset       = train_dataset.shuffle(600000, reshuffle_each_iteration=False)
        train_dataset       = train_dataset.batch(s_batch)

        if reload_weights==False:
            # Training
            history = self.fit(train_dataset, epochs=25)
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
            print("[INF] Restoring last model")
            status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
