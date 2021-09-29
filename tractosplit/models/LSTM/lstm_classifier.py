import tensorflow as tf
import matplotlib.pyplot as plt
import tractosplit.utils.constants as constants
from tractosplit.models.generators import SL_stochastic_generator, SL_basic_generator


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric], "")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, "val_" + metric])


# Small recurrent model
class lstmClassifier(tf.keras.Model):
    _emb_size = 32
    _rnn_size = 32
    _int_size = 32
    _s_batch = 12
    _epochs = 25

    def __init__(self):
        super(lstmClassifier, self).__init__(name="lstm_classifier")
        nclasses = constants.clusters["size"] + 1
        self.embedding = tf.keras.layers.Dense(self._emb_size)
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self._rnn_size))
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.dense = tf.keras.layers.Dense(self._int_size, activation="relu")
        self.final = tf.keras.layers.Dense(nclasses)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout1(x)
        x = self.lstm(x)
        x = self.dropout2(x)
        x = self.dense(x)
        x = self.final(x)
        return x

    def train(self, train_subjects, test_subjects, train_id):
        self.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=["accuracy"],
        )
        # Checkpoint
        checkpoint_dir = constants.lstm_path + train_id
        checkpoint_prefix = checkpoint_dir + constants.lstm_prefix
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)

        print("[INFO] Used for training:", train_subjects)
        print("[INFO] Used for testing:", test_subjects)
        # Training
        training_generator = SL_stochastic_generator(train_subjects)
        validation_generator = SL_basic_generator(test_subjects, batchsize=10000)
        history = self.fit(
            training_generator,
            epochs=self._epochs,
            validation_data=validation_generator,
        )
        checkpoint.save(file_prefix=checkpoint_prefix)
        # Plots
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plot_graphs(history, "accuracy")
        plt.ylim(None, 1)
        plt.subplot(1, 2, 2)
        plot_graphs(history, "loss")
        plt.ylim(0, None)
        plt.savefig(constants.train_report_path + train_id + "accuracy_loss.png")

    def restore(self, train_id):
        self.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=["accuracy"],
        )
        # Checkpoint
        checkpoint_dir = constants.lstm_path + train_id
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
        # Load parameters saved in previous trainings
        print("[INFO] Restoring lstm model:", train_id)
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        status.assert_existing_objects_matched()
        print("[INFO] Restored correctly")
