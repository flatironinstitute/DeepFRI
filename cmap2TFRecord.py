import pickle
import numpy as np
from utils import seq2onehot, load_catalogue
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
            A_nbr = cmap['A_nbr']
            S = seq2onehot(str(cmap['sequence']))
            L = cmap['L'].item()
            y = np.asarray(self.labels[chain], dtype=np.int64)

            d_feature = {}
            # load appropriate tf.train.Featur class depending on dtype
            d_feature['A_ca'] = tf.train.Feature(float_list=tf.train.FloatList(value=A_ca.reshape(-1)))
            d_feature['A_nbr'] = tf.train.Feature(float_list=tf.train.FloatList(value=A_nbr.reshape(-1)))
            d_feature['A_all'] = tf.train.Feature(float_list=tf.train.FloatList(value=A_all.reshape(-1)))
            d_feature['S'] = tf.train.Feature(float_list=tf.train.FloatList(value=S.reshape(-1)))

            d_feature['label'] = dtype_feature_y(y)
            d_feature['L'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[L]))

            features = tf.train.Features(feature=d_feature)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
        print ("Writing {} done!".format(tfrecord_fn))


if __name__ == "__main__":
    cmap2path = load_catalogue(fn='/mnt/ceph/users/vgligorijevic/ContactMaps/data/nr_pdb_chains/catalogue.csv')
    # cmap2path = load_catalogue()
    out_path = '/mnt/ceph/users/vgligorijevic/ContactMaps/data/TFRecord/'
    for seqid in [90]:
        path = '/mnt/home/vgligorijevic/Projects/NewMethods/Contact_maps/go_annot/pdb_GO_train_test_split_bc_' + str(seqid) + '.pckl'
        # path = '/mnt/ceph/users/vgligorijevic/ContactMaps/data/SIFTS_EC/pdb_EC_train_test_split_bc_' + str(seqid) + '.pckl'
        # path = '/mnt/ceph/users/vgligorijevic/ContactMaps/data/Swiss-Model/swiss-model_EC_train_test_split.pckl'
        annot = pickle.load(open(path, 'rb'))
        for ont in ['molecular_function', 'biological_process', 'cellular_component']:
            for split in ['train', 'valid']:
                chains = annot[ont][split + '_pdb_chains']
                Y = annot[ont]['Y_' + split]
                for num, evid in enumerate(['EXP', 'EXP-IEA']):
                    print ('pdb_chains_seqid_' + str(seqid) + '_' + ont + '_' + split + '_' + evid + '.tfrecords')
                    tmp_Y = Y.copy()
                    tmp_chains = chains.copy()

                    tmp_Y[tmp_Y == -1] = num
                    idx = np.where(tmp_Y.sum(axis=1) > 0)[0]
                    tmp_Y = tmp_Y[idx]
                    tmp_chains = chains[idx]

                    labels = {}
                    for i, chain in enumerate(tmp_chains):
                        if sum(tmp_Y[i]) > 0:
                            labels[chain] = tmp_Y[i]
                    tfr = GenerateTFRecord(labels)
                    tfr.convert_numpy_folder(cmap2path, out_path + 'pdb_chains_seqid_' + str(seqid) + '_' + ont + '_' + split + '_' + evid + '.tfrecords')
                    # tfr.convert_numpy_folder(cmap2path, out_path + 'swiss-model_chains_seqid_90_' + ont + '_' + split + '_EC.tfrecords')
