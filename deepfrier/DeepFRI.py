import tensorflow as tf

from .utils import micro_aupr, get_batched_dataset, EvaluateInputTensor

import keras.backend as K
from keras.models import Model, load_model
from keras import regularizers

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import Concatenate, Add
from keras.layers import Dense, Dropout, Input, Lambda, Activation
from .GCN_layer import GraphCNN

import matplotlib.pyplot as plt
plt.switch_backend('agg')


class DeepFRI(object):
    """
    Class containig the GCN + LM models for predicting protein function.
    """
    def __init__(self, output_dim, results_dir, n_channels=26, gcn_dims=[64, 128], hidd_dims=[512], lr=0.0002, drop=0.3, l2_reg=0.001, lm_model_name=None):
        """ Initialize the model
        :param output_dim: {int} number of GO terms/EC numbers
        :param results_dir: {string} directory to save all the results/ logs
        :param n_channels: {int} number of input features per residue (26 for 1-hot encoding)
        :param gcn_dims: {list <int>} number of hidden units in GCN layers
        :param hidd_dims: {list <int>} number of hiddne units in Dense layers
        :param lr: {float} learning rate for Adam optimizer
        :param drop: {float} dropout fraction for Dense layers
        :param l2_reg: {float} l2 regularization parameter
        :lm_model: {string} name of the pre-trained LSTM language model to be loaded
        """
        self.output_dim = output_dim
        self.results_dir = results_dir
        self.n_channels = n_channels
        self.gcn_dims = gcn_dims
        self.hidd_dims = hidd_dims
        self.lr = lr
        self.drop = drop
        self.l2_reg = l2_reg

        if lm_model_name is not None:
            lm_model = load_model(lm_model_name)
            self.lm_model = Model(inputs=lm_model.input, outputs=lm_model.get_layer("LSTM2").output)
            self.lm_model.trainable = False
        else:
            self.lm_model = None

    def _build_model(self, input_cmap, input_seq, output_label):
        x_aa = Dense(self.gcn_dims[0], use_bias=False, name='AA_embedding')(input_seq)
        if self.lm_model is not None:
            x_lm = Dense(self.gcn_dims[0], use_bias=True, name='LM_embedding')(self.lm_model(input_seq))
            x_aa = Add(name='Emedding')([x_lm, x_aa])
        x = Activation('relu')(x_aa)

        # Encoding layer
        gcnn_concat = []
        for l in range(0, len(self.gcn_dims)):
            x = GraphCNN(self.gcn_dims[l], use_bias=False, activation='relu',
                         kernel_regularizer=regularizers.l2(self.l2_reg), name='GCNN_' + str(l+1))([x, input_cmap])
            gcnn_concat.append(x)
        x = Concatenate(name='GCNN_concatenate')(gcnn_concat)

        # Sum pooling
        x = Lambda(lambda z: K.sum(z, axis=1), name='Sum_Pooling')(x)

        # Dense layers
        for l in range(0, len(self.hidd_dims)):
            x = Dense(units=self.hidd_dims[l], activation='relu')(x)
            x = Dropout((l+1)*self.drop)(x)

        x = Dense(units=self.output_dim, name='functions')(x)
        output_layer = Activation('sigmoid')(x)
        model = Model(inputs=[input_cmap, input_seq], outputs=output_layer)
        print (model.summary())

        optimizer = Adam(lr=self.lr, beta_1=0.95, beta_2=0.99)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc', micro_aupr], target_tensors=output_label)

        return model

    def _initialize_model(self, train_tfrecord_fn, valid_tfrecord_fn, batch_size=64, pad_len=1200, cmap_type='A_ca'):
        self.train_model = None
        self.valid_model = None
        # self.final_model = None

        # train model
        A_train_batch, S_train_batch, y_train_batch = get_batched_dataset([train_tfrecord_fn], batch_size=batch_size, pad_len=pad_len,
                                                                          n_goterms=self.output_dim, channels=self.n_channels, cmap_type=cmap_type)

        train_input_cmap = Input(tensor=A_train_batch, name='cmap')
        train_input_seq = Input(tensor=S_train_batch, name='seq')
        print ("### Compiling train model....")
        self.train_model = self._build_model(train_input_cmap, train_input_seq, [y_train_batch])

        # validation model
        A_valid_batch, S_valid_batch, y_valid_batch = get_batched_dataset([valid_tfrecord_fn], batch_size=batch_size, pad_len=pad_len,
                                                                          n_goterms=self.output_dim, channels=self.n_channels, cmap_type=cmap_type)
        valid_input_cmap = Input(tensor=A_valid_batch, name='cmap')
        valid_input_seq = Input(tensor=S_valid_batch, name='seq')
        print ("### Compiling valid model....")
        self.valid_model = self._build_model(valid_input_cmap, valid_input_seq, [y_valid_batch])

        # final model
        # A_batch, S_batch, y_batch = get_batched_dataset([train_tfrecord_fn, valid_tfrecord_fn], batch_size=batch_size, pad_len=pad_len,
        #                                                n_goterms=self.output_dim, channels=self.n_channels, cmap_type=cmap_type)

        # input_cmap = Input(tensor=A_batch, name='cmap')
        # input_seq = Input(tensor=S_batch, name='seq')

        # print ("### Compiling final model....")
        # self.final_model = self._build_model(input_cmap, input_seq, [y_batch])

    def train(self, train_tfrecord_fn, valid_tfrecord_fn, model_name_prefix, epochs=100, batch_size=64, pad_len=1200, cmap_type='A_ca'):
        self.model_name_prefix = model_name_prefix
        self._initialize_model(train_tfrecord_fn, valid_tfrecord_fn, batch_size=batch_size, pad_len=pad_len, cmap_type=cmap_type)

        # loading data
        n_train_records = sum(1 for _ in tf.python_io.tf_record_iterator(train_tfrecord_fn))
        n_valid_records = sum(1 for _ in tf.python_io.tf_record_iterator(valid_tfrecord_fn))
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
                                       callbacks=[EvaluateInputTensor(self.valid_model, steps=n_valid_records//batch_size), es, mc])
        self.history = history.history

        # fit final model
        # self.final_model.fit(epochs=len(self.history['loss']), steps_per_epoch=(n_train_records + n_valid_records)//batch_size)
        # self.final_model.save_weights(self.results_dir + self.model_name_prefix + '_best_train_model.h5')

        self._load_test_model()

    def _load_test_model(self):
        test_input_cmap = Input(shape=(None, None), name='cmap')
        test_input_seq = Input(shape=(None, self.n_channels), name='seq')
        print ("### Compiling test model...")
        self.test_model = self._build_model(test_input_cmap, test_input_seq, None)
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
        plt.plot(self.history['micro_aupr'], '-')
        plt.plot(self.history['val_micro_aupr'], '-')
        plt.title('model AUPR')
        plt.ylabel('micro-AUPR')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.results_dir + self.model_name_prefix + '_model_accuracy.png', bbox_inches='tight')

    def save_model(self):
        self.test_model.save(self.results_dir + self.model_name_prefix + '.h5')
