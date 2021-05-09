import glob
import tensorflow as tf

from .utils import get_batched_dataset
from .layers import FuncPredictor, SumPooling
from .layers import ChebConv, GraphConv, SAGEConv, MultiGraphConv, NoGraphConv, GAT

import warnings
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class DeepFRI(object):
    """ Class containig the GCN + LM models for predicting protein function. """
    def __init__(self, output_dim, n_channels=26, gc_dims=[64, 128], fc_dims=[512], lr=0.0002, drop=0.3, l2_reg=1e-4,
                 gc_layer=None, lm_model_name=None, model_name_prefix=None):
        """ Initialize the model
        :param output_dim: {int} number of GO terms/EC numbers
        :param n_channels: {int} number of input features per residue (26 for 1-hot encoding)
        :param gc_dims: {list <int>} number of hidden units in GConv layers
        :param fc_dims: {list <int>} number of hiddne units in Dense layers
        :param lr: {float} learning rate for Adam optimizer
        :param drop: {float} dropout fraction for Dense layers
        :param gc_layer: {str} Graph Convolution layer
        :lm_model: {string} name of the pre-trained LSTM language model to be loaded
        :model_name_prefix: {string} name of a deepFRI model to be saved
        """
        self.output_dim = output_dim
        self.n_channels = n_channels
        self.model_name_prefix = model_name_prefix

        if lm_model_name is not None:
            lm_model = tf.keras.models.load_model(lm_model_name)
            lm_model = tf.keras.Model(inputs=lm_model.input,
                                      outputs=tf.keras.layers.Concatenate()([lm_model.get_layer("LSTM1").output, lm_model.get_layer("LSTM2").output]))
            lm_model.trainable = False
        else:
            lm_model = None

        # build and compile model
        self._build_model(gc_dims, fc_dims, n_channels, output_dim, lr, drop, l2_reg, gc_layer, lm_model=lm_model)

    def _build_model(self, gc_dims, fc_dims, n_channels, output_dim, lr, drop, l2_reg, gc_layer=None, lm_model=None):

        if gc_layer == 'NoGraphConv':
            self.GConv = NoGraphConv
            self.gc_layer = gc_layer
        elif gc_layer == 'GAT':
            self.GConv = GAT
            self.gc_layer = gc_layer
        elif gc_layer == 'GraphConv':
            self.GConv = GraphConv
            self.gc_layer = gc_layer
        elif gc_layer == 'MultiGraphConv':
            self.GConv = MultiGraphConv
            self.gc_layer = gc_layer
        elif gc_layer == 'SAGEConv':
            self.GConv = SAGEConv
            self.gc_layer = gc_layer
        elif gc_layer == 'ChebConv':
            self.GConv = ChebConv
            self.gc_layer = gc_layer
        else:
            self.GConv = NoGraphConv
            self.gc_layer = 'NoGraphConv'
            warnings.warn('gc_layer not specified! No GraphConv used!')

        print ("### Compiling DeepFRI model with %s layer..." % (gc_layer))

        input_cmap = tf.keras.layers.Input(shape=(None, None), name='cmap')
        input_seq = tf.keras.layers.Input(shape=(None, n_channels), name='seq')

        # Encoding layers
        lm_dim = 1024
        x_aa = tf.keras.layers.Dense(lm_dim, use_bias=False, name='AA_embedding')(input_seq)
        if lm_model is not None:
            x_lm = tf.keras.layers.Dense(lm_dim, use_bias=True, name='LM_embedding')(lm_model(input_seq))
            x_aa = tf.keras.layers.Add(name='Embedding')([x_lm, x_aa])
        x = tf.keras.layers.Activation('relu')(x_aa)

        # Graph Convolution layer
        gcnn_concat = []
        for l in range(0, len(gc_dims)):
            x = self.GConv(gc_dims[l], use_bias=False, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                           name=self.gc_layer + '_' + str(l+1))([x, input_cmap])
            gcnn_concat.append(x)

        if len(gcnn_concat) > 1:
            x = tf.keras.layers.Concatenate(name='GCNN_concatenate')(gcnn_concat)
        else:
            x = gcnn_concat[-1]

        # Sum pooling
        x = SumPooling(axis=1, name='SumPooling')(x)

        # Dense layers
        for l in range(0, len(fc_dims)):
            x = tf.keras.layers.Dense(units=fc_dims[l], activation='relu')(x)
            x = tf.keras.layers.Dropout((l + 1)*drop)(x)

        # Output layer
        output_layer = FuncPredictor(output_dim=output_dim, name='labels')(x)

        self.model = tf.keras.Model(inputs=[input_cmap, input_seq], outputs=output_layer)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.95, beta_2=0.99)
        pred_loss = tf.keras.losses.CategoricalCrossentropy()
        self.model.compile(optimizer=optimizer, loss=pred_loss, metrics=['acc'])
        print (self.model.summary())

    def train(self, train_tfrecord_fn, valid_tfrecord_fn,
              epochs=100, batch_size=64, pad_len=1200, cmap_type='ca', cmap_thresh=10.0, ont='mf', class_weight=None):

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
                                          cmap_type=cmap_type,
                                          cmap_thresh=cmap_thresh,
                                          ont=ont)

        # validation tfrecords
        batch_valid = get_batched_dataset(valid_tfrecord_fn,
                                          batch_size=batch_size,
                                          pad_len=pad_len,
                                          n_goterms=self.output_dim,
                                          channels=self.n_channels,
                                          cmap_type=cmap_type,
                                          cmap_thresh=cmap_thresh,
                                          ont=ont)

        # early stopping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        # model checkpoint
        mc = tf.keras.callbacks.ModelCheckpoint(self.model_name_prefix + '_best_train_model.h5', monitor='val_loss', mode='min', verbose=1,
                                                save_best_only=True, save_weights_only=True)

        # fit model
        history = self.model.fit(batch_train,
                                 epochs=epochs,
                                 validation_data=batch_valid,
                                 steps_per_epoch=n_train_records//batch_size,
                                 validation_steps=n_valid_records//batch_size,
                                 class_weight=class_weight,
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
        self.model = tf.keras.models.load_model(self.model_name_prefix + '.hdf5',
                                                custom_objects={self.gc_layer: self.GConv,
                                                                'FuncPredictor': FuncPredictor,
                                                                'SumPooling': SumPooling
                                                                })
