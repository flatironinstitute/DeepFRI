import sys
import pickle
import numpy as np
from deepfrier.utils import seq2onehot, load_catalogue, norm_adj
import tensorflow as tf


class GenerateTFRecord(object):
    def __init__(self, labels):
        self.labels = labels

    def _dtype_feature(self):
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))

    def convert_numpy_folder(self, catalogue, tfrecord_fn):
        writer = tf.python_io.TFRecordWriter(tfrecord_fn)
        print ("Serializing {:d} examples into {}".format(len(self.labels), tfrecord_fn))

        dtype_feature_y = self._dtype_feature()
        for chain in self.labels:
            cmap = np.load(catalogue[chain])
            A_ca = cmap['A_ca_10A']
            A_all = cmap['A_all_6A']
            if 'A_nbr' in cmap:
                A_nbr = cmap['A_nbr']
            elif 'A_nbr_bin' in cmap:
                A_nbr = norm_adj(cmap['A_nbr_bin'])
            else:
                sys.exit("NBR contact maps not in the *npz file!")
            S = seq2onehot(str(cmap['sequence']))
            L = cmap['L'].item()
            y = np.asarray(self.labels[chain], dtype=np.int64)

            d_feature = {}
            # load appropriate tf.train.Featur class depending on dtype
            d_feature['A_ca'] = tf.train.Feature(float_list=tf.train.FloatList(value=A_ca.reshape(-1)))
            d_feature['A_nbr'] = tf.train.Feature(float_list=tf.train.FloatList(value=A_nbr.reshape(-1)))
            d_feature['A_all'] = tf.train.Feature(float_list=tf.train.FloatList(value=A_all.reshape(-1)))
            d_feature['S'] = tf.train.Feature(float_list=tf.train.FloatList(value=S.reshape(-1)))

            d_feature['labels'] = dtype_feature_y(y)
            d_feature['L'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[L]))

            features = tf.train.Features(feature=d_feature)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
        print ("Writing {} done!".format(tfrecord_fn))


if __name__ == "__main__":
    cmap2path = load_catalogue(fn='/mnt/ceph/users/vgligorijevic/ContactMaps/data/Swiss-Model/merged_annot/catalogue.csv')
    out_path = '/mnt/ceph/users/vgligorijevic/ContactMaps/data/Swiss-Model/merged_annot/tfrecords/'
    pckl_fn = '/mnt/ceph/users/vgligorijevic/ContactMaps/data/Swiss-Model/merged_annot/merged_GO_train_test_split_nr30.pckl'
    num = {'EXP': 0, 'EXP-IEA': 1}

    split = str(sys.argv[1])
    evid = str(sys.argv[2])
    cmap = str(sys.argv[3])
    ont = str(sys.argv[4])
    begin = int(sys.argv[5])
    end = int(sys.argv[6])

    annot = pickle.load(open(pckl_fn, 'rb'))
    ontology = annot['ontology']
    Y = annot['Y_' + evid].toarray()
    go_idx = np.where(ontology == ont)[0]
    Y = Y[:, go_idx]
    Y[Y == -1] = num[evid]

    chains = annot[split + '_pdb_chains']
    cmap_type = annot[split + '_cmap_type']
    idx = np.where(cmap_type == cmap)[0]
    chains = chains[idx]
    Y = Y[idx]

    labels = {}
    for i, chain in enumerate(chains):
        labels[chain] = Y[i]

    tfr = GenerateTFRecord(labels)
    tfr.convert_numpy_folder(cmap2path,
                             out_path + cmap + '_chains_' + ont + '_seqid_30_' + split + '_' + evid + '_' + str(begin) + '-' + str(end) + '.tfrecords')
