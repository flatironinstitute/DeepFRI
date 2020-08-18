import glob
import tensorflow as tf

from .utils import get_batched_dataset
from .layers import FuncPredictor

import matplotlib.pyplot as plt
plt.switch_backend('agg')


class DeepCNN(object):
    """ Class containig the CNN model for predicting protein function. """
    def __init__(self, output_dim, n_channels=26, num_filters=[100], filter_lens=[3], lr=0.0002, drop=0.3, l2_reg=0.001,
                 lm_model_name=None, model_name_prefix=None):
        """ Initialize the model
        :param output_dim: {int} number of GO terms/EC numbers
        :param n_channels: {int} number of input features per residue (26 for 1-hot encoding)
        :param num_filters: {list <int>} number of  filters in the first CNN layer
        :param filter_lens: {list <int>} filter lengths in the first CNN layer
        :param lr: {float} learning rate for Adam optimizer
        :param drop: {float} dropout fraction for Dense layers
        :param l2_reg: {float} l2 regularization parameter
        :lm_model: {string} name of the pre-trained LSTM language model to be loaded
        """
        self.output_dim = output_dim
        self.n_channels = n_channels
        self.model_name_prefix = model_name_prefix

        if lm_model_name is not None:
            lm_model = tf.keras.models.load_model(lm_model_name)
            lm_model = tf.keras.Model(inputs=lm_model.input, outputs=lm_model.get_layer("LSTM2").output)
            lm_model.trainable = False
        else:
            lm_model = None

        # build and compile model
        self._build_model(num_filters, filter_lens, n_channels, output_dim, lr, drop, l2_reg, lm_model=lm_model)

    def _build_model(self, num_filters, filter_lens, n_channels, output_dim, lr, drop, l2_reg, lm_model=None):
        print ("### Compiling DeepCNN model...")

        input_seq = tf.keras.layers.Input(shape=(None, n_channels), name='seq')

        # Encoding layers
        x = input_seq
        if lm_model is not None:
            x_lm = tf.keras.layers.Dense(128, use_bias=False, name='LM_embedding')(lm_model(x))
            x_aa = tf.keras.layers.Dense(128, name='AA_embedding')(x)
            x = tf.keras.layers.Add(name='Emedding')([x_lm, x_aa])
            x = tf.keras.layers.Activation('relu')(x)

        # Conv layers
        x_concat = []
        for l in range(0, len(num_filters)):
            x_l = tf.keras.layers.Conv1D(filters=num_filters[l], kernel_size=filter_lens[l],
                                         padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
            x_concat.append(x_l)

        x = tf.keras.layers.Concatenate()(x_concat)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu', name='CNN_concatenate')(x)
        x = tf.keras.layers.Dropout(drop)(x)

        # 1-d features
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dropout(2*drop)(x)

        # Output layer
        output_layer = FuncPredictor(output_dim=output_dim, name='labels')(x)

        self.model = tf.keras.Model(inputs=input_seq, outputs=output_layer)
        optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=0.95, beta_2=0.99)
        pred_loss = tf.keras.losses.CategoricalCrossentropy()
        self.model.compile(optimizer=optimizer, loss=pred_loss, metrics=['acc'])
        print (self.model.summary())

    def train(self, train_tfrecord_fn, valid_tfrecord_fn, epochs=100, batch_size=64, pad_len=1000, ont='mf', class_weight=None):
        n_train_records = sum(1 for f in glob.glob(train_tfrecord_fn) for _ in tf.data.TFRecordDataset(f))
        n_valid_records = sum(1 for f in glob.glob(valid_tfrecord_fn) for _ in tf.data.TFRecordDataset(f))
        print ("### Training on: ", n_train_records, "contact maps.")
        print ("### Validating on: ", n_valid_records, "contact maps.")

        # train tfrecords
        batch_train = get_batched_dataset(train_tfrecord_fn,
                                          batch_size=batch_size,
                                          pad_len=pad_len,
                                          n_goterms=self.output_dim,
                                          channels=self.n_channels,
                                          gcn=False,
                                          ont=ont)

        # validation tfrecords
        batch_valid = get_batched_dataset(valid_tfrecord_fn,
                                          batch_size=batch_size,
                                          pad_len=pad_len,
                                          n_goterms=self.output_dim,
                                          channels=self.n_channels,
                                          gcn=False,
                                          ont=ont)

        # early stopping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)

        # model checkpoint
        mc = tf.keras.callbacks.ModelCheckpoint(self.model_name_prefix + '_best_train_model.h5', monitor='val_loss', mode='min', verbose=2,
                                                save_best_only=True, save_weights_only=True)

        # fit model
        history = self.model.fit(batch_train,
                                 epochs=epochs,
                                 validation_data=batch_valid,
                                 class_weight=class_weight,
                                 steps_per_epoch=n_train_records//batch_size,
                                 validation_steps=n_valid_records//batch_size,
                                 callbacks=[es, mc])

        self.history = history.history

    def predict(self, input_data):
        return self.model(input_data).numpy()[0][:, 0]

    def plot_losses(self):
        plt.figure()
        plt.plot(self.history['loss'], '-')
        plt.plot(self.history['val_loss'], '-')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.model_name_prefix + '_model_loss.png', bbox_inches='tight')

        plt.figure()
        plt.plot(self.history['acc'], '-')
        plt.plot(self.history['val_acc'], '-')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.model_name_prefix + '_model_accuracy.png', bbox_inches='tight')

    def save_model(self):
        self.model.save(self.model_name_prefix + '.hdf5')

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_name_prefix,
                                                custom_objects={'FuncPredictor': FuncPredictor})
