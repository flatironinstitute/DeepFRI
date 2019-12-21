import tensorflow as tf

from .utils import get_batched_dataset, EvaluateInputTensor

from keras.models import Model, load_model
from keras import regularizers

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import Concatenate, Add, Conv1D
from keras.layers import BatchNormalization
from keras.layers import Reshape
from keras.layers import Dense, Dropout, Input, Activation, GlobalMaxPooling1D

import matplotlib.pyplot as plt
plt.switch_backend('agg')


class DeepCNN(object):
    """
    Class containig the GCN + LM models for predicting protein function.
    """
    def __init__(self, output_dim, results_dir, n_channels=26, num_filters=[100], filter_lens=[3], lr=0.0002, drop=0.3, l2_reg=0.001, lm_model_name=None):
        """ Initialize the model
        :param output_dim: {int} number of GO terms/EC numbers
        :param results_dir: {string} directory to save all the results/ logs
        :param n_channels: {int} number of input features per residue (26 for 1-hot encoding)
        :param num_filters: {list <int>} number of  filters in the first CNN layer
        :param filter_lens: {list <int>} filter lengths in the first CNN layer
        :param lr: {float} learning rate for Adam optimizer
        :param drop: {float} dropout fraction for Dense layers
        :param l2_reg: {float} l2 regularization parameter
        :lm_model: {string} name of the pre-trained LSTM language model to be loaded
        """
        self.output_dim = output_dim
        self.results_dir = results_dir
        self.n_channels = n_channels
        self.num_filters = num_filters
        self.filter_lens = filter_lens
        self.lr = lr
        self.drop = drop
        self.l2_reg = l2_reg

        if lm_model_name is not None:
            lm_model = load_model(lm_model_name)
            self.lm_model = Model(inputs=lm_model.input, outputs=lm_model.get_layer("LSTM2").output)
            self.lm_model.trainable = False
        else:
            self.lm_model = None

    def _build_model(self, input_seq, output_label):
        x = input_seq
        if self.lm_model is not None:
            x_lm = Dense(64, use_bias=False, name='LM_embedding')(self.lm_model(x))
            x_aa = Dense(64, name='AA_embedding')(x)
            x = Add(name='Emedding')([x_lm, x_aa])
            x = Activation('relu')(x)

        x_concat = []
        # Encoding layers
        for l in range(0, len(self.num_filters)):
            x_l = Conv1D(filters=self.num_filters[l], kernel_size=self.filter_lens[l], padding='same', kernel_regularizer=regularizers.l2(self.l2_reg))(x)
            x_concat.append(x_l)
        x = Concatenate()(x_concat)
        x = BatchNormalization()(x)
        x = Activation('relu', name='CNN_concatenate')(x)
        # x = Dropout(self.drop)(x)

        # CNN layers
        # x = Conv1D(filters=int(2*sum(self.num_filters)), kernel_size=3, padding='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu', name='CNN_layer')(x)

        # 1-d features
        x = GlobalMaxPooling1D()(x)
        x = Dropout(2*self.drop)(x)

        # Output layer
        output_layers = []
        for i in range(self.output_dim):
            x_out = Dense(units=2, activation='softmax', name='goterm_' + str(i+1))(x)
            x_out = Reshape(target_shape=(1, 2))(x_out)
            output_layers.append(x_out)
        output_layer = Concatenate(axis=1, name='functions')(output_layers)

        # output_layer = Dense(self.output_dim, activation='sigmoid', name='functions')(x)

        model = Model(inputs=input_seq, outputs=output_layer)
        print (model.summary())

        optimizer = Adam(lr=self.lr, beta_1=0.95, beta_2=0.99)
        # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'], target_tensors=output_label)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'], target_tensors=output_label)

        return model

    def _initialize_model(self, train_tfrecord_fn, valid_tfrecord_fn, batch_size=64, pad_len=1200):
        self.train_model = None
        self.valid_model = None
        # self.final_model = None

        # train model
        S_train_batch, y_train_batch = get_batched_dataset(train_tfrecord_fn, batch_size=batch_size, pad_len=pad_len,
                                                           n_goterms=self.output_dim, channels=self.n_channels, gcn=False)

        train_input_seq = Input(tensor=S_train_batch, name='seq')
        self.train_model = self._build_model(train_input_seq, [y_train_batch])

        # validation model
        S_valid_batch, y_valid_batch = get_batched_dataset(valid_tfrecord_fn, batch_size=batch_size, pad_len=pad_len,
                                                           n_goterms=self.output_dim, channels=self.n_channels, gcn=False)

        valid_input_seq = Input(tensor=S_valid_batch, name='seq')
        self.valid_model = self._build_model(valid_input_seq, [y_valid_batch])

        # final model
        # S_batch, y_batch = get_batched_dataset(train_tfrecord_fn, valid_tfrecord_fn, batch_size=batch_size, pad_len=pad_len,
        #                                        n_goterms=self.output_dim, channels=self.n_channels, gcn=False)

        # input_seq = Input(tensor=S_batch, name='seq')
        # self.final_model = self._build_model(input_seq, [y_batch])

    def train(self, train_tfrecord_fn, valid_tfrecord_fn, model_name_prefix, epochs=100, batch_size=64, pad_len=1200, class_weight=None):
        self.model_name_prefix = model_name_prefix
        self._initialize_model(train_tfrecord_fn, valid_tfrecord_fn, batch_size=batch_size, pad_len=pad_len)

        # loading data
        n_train_records = sum(1 for f in train_tfrecord_fn for _ in tf.python_io.tf_record_iterator(f))
        n_valid_records = sum(1 for f in valid_tfrecord_fn for _ in tf.python_io.tf_record_iterator(f))
        # n_train_records = sum(1 for _ in tf.python_io.tf_record_iterator(train_tfrecord_fn))
        # n_valid_records = sum(1 for _ in tf.python_io.tf_record_iterator(valid_tfrecord_fn))
        print ("### Training on: ", n_train_records, "contact maps.")
        print ("### Validating on: ", n_valid_records, "contact maps.")

        # early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        # model checkpoint
        mc = ModelCheckpoint(self.results_dir + self.model_name_prefix + '_best_train_model.h5', monitor='val_loss', mode='min', verbose=1,
                             save_best_only=True, save_weights_only=True)

        # fit model
        history = self.train_model.fit(epochs=epochs,
                                       steps_per_epoch=n_train_records//batch_size,
                                       class_weight=class_weight,
                                       callbacks=[EvaluateInputTensor(self.valid_model, steps=n_valid_records//batch_size), es, mc])
        self.history = history.history

        # fit final model
        # self.final_model.fit(epochs=len(self.history['loss']), steps_per_epoch=(n_train_records + n_valid_records)//batch_size)
        # self.final_model.save_weights(self.results_dir + self.model_name_prefix + '_best_train_model.h5')

        self._load_test_model()

    def _load_test_model(self):
        test_input_seq = Input(shape=(None, self.n_channels), name='seq')
        self.test_model = self._build_model(test_input_seq, None)
        self.test_model.load_weights(self.results_dir + self.model_name_prefix + '_best_train_model.h5')

    def predict(self, input_data):
        return self.test_model.predict(input_data)

    def plot_losses(self):
        plt.figure()
        plt.plot(self.history['loss'], '-')
        plt.plot(self.history['val_loss'], '-')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.results_dir + self.model_name_prefix + '_model_loss.png', bbox_inches='tight')

        plt.figure()
        plt.plot(self.history['acc'], '-')
        plt.plot(self.history['val_acc'], '-')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.results_dir + self.model_name_prefix + '_model_accuracy.png', bbox_inches='tight')

    def save_model(self):
        self.test_model.save(self.results_dir + self.model_name_prefix + '.hdf5')
