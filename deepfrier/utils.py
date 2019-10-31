import csv
import numpy as np
import networkx as nx
import tensorflow as tf
from keras.callbacks import Callback
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score

from Bio import SeqIO


def load_FASTA(filename):
    # Loads fasta file and returns a list of the Bio SeqIO records
    infile = open(filename, 'rU')
    entries = []
    proteins = []
    for entry in SeqIO.parse(infile, 'fasta'):
        entries.append(str(entry.seq))
        proteins.append(str(entry.id))
    if(len(entries) == 0):
        return False
    return proteins, entries


def rnd_adj(N, p):
    G = nx.erdos_renyi_graph(N, p)
    A = norm_adj(nx.adjacency_matrix(G).toarray().astype(float))

    return A


def norm_adj(A, symm=True):
    #  Normalize adj matrix
    if symm:
        A += np.eye(A.shape[1])
        d = 1.0/np.sqrt(A.sum(axis=1))
        d = np.diag(d)
        A = d.dot(A.dot(d))
    else:
        A /= A.sum(axis=1)[:, np.newaxis]

    return A


def get_thresholds(Y_test, Y_hat_test):
    thresholds = []
    accuracies = []
    f1_scores = []
    for i in range(Y_test.shape[1]):
        pre, rec, thresh = precision_recall_curve(Y_test[:, i], Y_hat_test[:, i])
        pre = pre[:-1]
        rec = rec[:-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2*pre*rec/(pre + rec)
        f1[np.isnan(f1)] = 0
        idx = np.argmax(f1)
        y_hat = (Y_hat_test[:, i] > thresh[idx]).astype(int)
        accuracies.append(accuracy_score(Y_test[:, i], y_hat))
        f1_scores.append(f1[idx])
        thresholds.append(thresh[idx])

    return np.asarray(thresholds), np.asarray(f1_scores), np.asarray(accuracies)


def _micro_aupr(y_true, y_test):
    return average_precision_score(y_true, y_test, average='micro')


def micro_aupr(y_true, y_pred):
    return tf.py_func(_micro_aupr, (y_true, y_pred), tf.double)


class EvaluateInputTensor(Callback):
    """ Validate a model which does not expect external numpy data during training.

    Keras does not expect external numpy data at training time, and thus cannot
    accept numpy arrays for validation when all of a Keras Model's
    `Input(input_tensor)` layers are provided an  `input_tensor` parameter,
    and the call to `Model.compile(target_tensors)` defines all `target_tensors`.
    Instead, create a second model for validation which is also configured
    with input tensors and add it to the `EvaluateInputTensor` callback
    to perform validation.

    It is recommended that this callback be the first in the list of callbacks
    because it defines the validation variables required by many other callbacks,
    and Callbacks are made in order.

    # Arguments
        model: Keras model on which to call model.evaluate().
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
    """

    def __init__(self, model, steps, metrics_prefix='val', verbose=1):
        # parameter of callbacks passed during initialization
        # pass evalation mode directly
        super(EvaluateInputTensor, self).__init__()
        self.val_model = model
        self.num_steps = steps
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix

    def on_epoch_end(self, epoch, logs={}):
        self.val_model.set_weights(self.model.get_weights())
        results = self.val_model.evaluate(None, None, steps=int(self.num_steps),
                                          verbose=self.verbose)
        metrics_str = '\n'
        for result, name in zip(results, self.val_model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)


def seq2onehot(seq):
    """Create 26-dim embedding"""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x


def _parse_function_gcn(serialized, n_goterms, channels, cmap_type='A_ca'):
    features = {
        cmap_type: tf.VarLenFeature(dtype=tf.float32),
        "S": tf.VarLenFeature(dtype=tf.float32),
        "label": tf.FixedLenFeature([n_goterms], dtype=tf.int64),
        "L": tf.FixedLenFeature([1], dtype=tf.int64)
    }
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)

    # Get all data
    L = parsed_example['L']

    A_shape = tf.stack([L[0], L[0]])
    A = parsed_example[cmap_type]
    A = tf.cast(A, tf.float32)
    A = tf.sparse_tensor_to_dense(A)
    A = tf.reshape(A, A_shape)

    S_shape = tf.stack([L[0], channels])
    S = parsed_example['S']
    S = tf.cast(S, tf.float32)
    S = tf.sparse_tensor_to_dense(S)
    S = tf.reshape(S, S_shape)

    y = parsed_example['label']
    y = tf.cast(y, tf.float32)

    return A, S, y


def _parse_function_cnn(serialized, n_goterms, channels):
    features = {
        "S": tf.VarLenFeature(dtype=tf.float32),
        "label": tf.FixedLenFeature([n_goterms], dtype=tf.int64),
        "L": tf.FixedLenFeature([1], dtype=tf.int64)
    }
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)

    # Get all data
    L = parsed_example['L']

    S_shape = tf.stack([L[0], channels])
    S = parsed_example['S']
    S = tf.cast(S, tf.float32)
    S = tf.sparse_tensor_to_dense(S)
    S = tf.reshape(S, S_shape)

    y = parsed_example['label']
    y = tf.cast(y, tf.float32)

    return S, y


def get_batched_dataset(filenames, batch_size=64, pad_len=1200, n_goterms=347, channels=26, gcn=True, cmap_type='A_ca'):

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    # dataset = dataset.map(lambda x: _parse_function(x, n_goterms=n_goterms, channels=channels), num_parallel_calls=5)
    if gcn:
        dataset = dataset.map(lambda x: _parse_function_gcn(x, n_goterms=n_goterms, channels=channels, cmap_type=cmap_type))
    else:
        dataset = dataset.map(lambda x: _parse_function_cnn(x, n_goterms=n_goterms, channels=channels))

    # Randomizes input using a window of 512 elements (read into memory)
    dataset = dataset.shuffle(buffer_size=128)
    dataset = dataset.repeat()  # Repeats dataset this # times
    # dataset = dataset.batch(batch_size)  # Batch size to use
    if gcn:
        dataset = dataset.padded_batch(batch_size, padded_shapes=([pad_len, pad_len], [pad_len, channels], [None]))
    else:
        dataset = dataset.padded_batch(batch_size, padded_shapes=([pad_len, channels], [None]))

    iterator = dataset.make_one_shot_iterator()
    # next_S, next_y = iterator.get_next()
    # next_A, next_S, next_y = iterator.get_next()
    batch = iterator.get_next()

    return batch


def load_catalogue(fn='/mnt/home/dberenberg/ceph/SWISSMODEL_CONTACTMAPS/catalogue.csv'):
    chain2path = {}
    with open(fn) as tsvfile:
        fRead = csv.reader(tsvfile, delimiter=',')
        # next(fRead, None)
        for line in fRead:
            # pdb = line[0].strip()
            # chain = line[1].strip()
            # path = line[3].strip()
            pdb_chain = line[0].strip()
            path = line[1].strip()
            # chain2path[pdb + '-' + chain] = path
            chain2path[pdb_chain] = path
    return chain2path
