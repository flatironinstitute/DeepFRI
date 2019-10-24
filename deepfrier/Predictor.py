import pickle
import numpy as np
import keras.backend as K

from keras.models import load_model
from utils import micro_aupr, load_catalogue, seq2onehot
from .GCN_layer import GraphCNN


class Predictor(object):
    """
    Class for loading trained models and computing GO/EC predictions and class activation maps (CAMs).
    """
    def __init__(self, model_prefix, output_fn_prefix, results_dir='./results/', gcn=True):
        self.model_prefix = model_prefix
        self.output_fn_prefix = output_fn_prefix
        self.results_dir = results_dir
        self.gcn = gcn
        self._load_model()

    def _load_model(self):
        self.model = load_model(self.results_dir + self.model_prefix + '.h5', custom_objects={'GraphCNN': GraphCNN, 'micro_aupr': micro_aupr})
        metadata = pickle.load(open(self.results_dir + self.model_prefix + '_thresholds.pckl', 'rb'))
        self.thresh = metadata['thresh']
        self.gonames = metadata['gonames']
        self.goterms = metadata['goterms']

    def predict(self, catalogue_fn, test_prot_list):
        print ("### Computing predictions...")
        self.chain2path = load_catalogue(catalogue_fn)
        self.Y_hat = np.zeros((len(test_prot_list), len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        if self.gcn:
            for i, chain in enumerate(test_prot_list):
                cmap = np.load(self.chain2path[chain])
                A = cmap['A_ca_10A']
                S = seq2onehot(str(cmap['sequence']))
                S = S.reshape(1, *S.shape)
                A = A.reshape(1, *A.shape)
                y = self.model.predict([A, S])
                self.Y_hat[i] = y
                go_idx = np.where((y.reshape(-1) >= self.thresh) == True)[0]
                for idx in go_idx:
                    if idx not in self.goidx2chains:
                        self.goidx2chains[idx] = set()
                    self.goidx2chains[idx].add(chain)
        else:
            for i, chain in enumerate(test_prot_list):
                cmap = np.load(self.chain2path[chain])
                S = seq2onehot(str(cmap['sequence']))
                S = S.reshape(1, *S.shape)
                y = self.model.predict(S)
                self.Y_hat[i] = y
                go_idx = np.where((y.reshape(-1) >= self.thresh) == True)[0]
                for idx in go_idx:
                    if idx not in self.goidx2chains:
                        self.goidx2chains[idx] = set()
                    self.goidx2chains[idx].add(chain)

    def save_predictions(self):
        print ("### Saving predictions to *.pckl file...")
        pickle.dump({'goidx2chains': self.goidx2chains, 'Y_hat': self.Y_hat, 'goterms': self.goterms, 'gonames': self.gonames},
                    open(self.results_dir + self.output_fn_prefix + '_predictions.pckl' + '', 'wb'))

    def _gradCAM(self, input_data_list, class_idx, layer_name='GCNN_layer'):
        class_output = self.model.output[:, class_idx]
        last_conv_layer = self.model.get_layer(layer_name)

        grads = K.gradients(class_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=1)

        if self.gcn:
            iterate = K.function([*self.model.input], [pooled_grads, last_conv_layer.output])
        else:
            iterate = K.function([self.model.input], [pooled_grads, last_conv_layer.output])

        heatmaps = []
        for x_input in input_data_list:
            pooled_grads_value, conv_layer_output_value = iterate(x_input)
            num_filters = pooled_grads_value.shape[-1]
            num_samples = pooled_grads_value.shape[0]
            for i in range(num_samples):
                for j in range(num_filters):
                    conv_layer_output_value[i, :, j] *= pooled_grads_value[i, j]

            heatmap = np.mean(conv_layer_output_value, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap, axis=-1)[:, np.newaxis]
            heatmap[np.isnan(heatmap)] = 0.0
            heatmaps.append(heatmap)

        return heatmaps

    def compute_gradCAM(self, layer_name='GCNN_layer'):
        self.pdb2cam = {}
        for go_indx in self.goidx2chains:
            input_data = []
            pred_chains = list(self.goidx2chains[go_indx])
            print ("### Computing gradCAM for ", self.gonames[go_indx], '... [# proteins=', len(pred_chains), ']')
            for chain in pred_chains:
                cmap = np.load(self.chain2path[chain])
                seq = str(cmap['sequence'])
                S = seq2onehot(seq)
                S = S.reshape(1, *S.shape)
                if self.gcn:
                    A = cmap['A_ca_10A']
                    A = A.reshape(1, *A.shape)
                    input_data.append([A, S])
                else:
                    input_data.append([S])
                if chain not in self.pdb2cam:
                    self.pdb2cam[chain] = {}
                    self.pdb2cam[chain]['GO_ids'] = []
                    self.pdb2cam[chain]['GO_names'] = []
                    self.pdb2cam[chain]['sequence'] = None
                    self.pdb2cam[chain]['saliency_maps'] = []
                self.pdb2cam[chain]['GO_ids'].append(self.goterms[go_indx])
                self.pdb2cam[chain]['GO_names'].append(self.gonames[go_indx])
                self.pdb2cam[chain]['sequence'] = seq
            heatmaps = self._gradCAM(input_data, go_indx, layer_name=layer_name)
            for i, chain in enumerate(pred_chains):
                self.pdb2cam[chain]['saliency_maps'].append(heatmaps[i])

    def save_gradCAM(self):
        print ("### Saving CAMs to *.pckl file...")
        pickle.dump(self.pdb2cam, open(self.results_dir + self.output_fn_prefix + "_saliency_maps.pckl", 'wb'))


if __name__ == "__main__":
    catalogue = '/mnt/ceph/users/vgligorijevic/ContactMaps/data/nr_pdb_chains/catalogue.csv'
    model_name = 'GCN-LM_SWISS-molecular_function_EXP-IEA_seqid_90_gcn_128-256-512_hidd_750'
    annot = pickle.load(open('/mnt/home/vgligorijevic/Projects/NewMethods/Contact_maps/go_annot/pdb_GO_train_test_split_bc_30.pckl', 'rb'))
    test_chains = annot['molecular_function']['test_pdb_chains']

    fri = Predictor(model_name, 'GCN_test_CAM', results_dir='../results/', gcn=True)

    fri.predict(catalogue, test_chains)
    fri.save_predictions()

    fri.compute_gradCAM(layer_name='GCNN_concatenate')
    fri.save_gradCAM()
