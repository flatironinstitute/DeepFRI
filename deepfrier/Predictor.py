import csv
import pickle
import numpy as np
import keras.backend as K

from keras.models import load_model
from .utils import micro_aupr, load_catalogue, load_FASTA, seq2onehot
from .GCN_layer import GraphCNN


class Predictor(object):
    """
    Class for loading trained models and computing GO/EC predictions and class activation maps (CAMs).
    """
    def __init__(self, model_prefix, gcn=True):
        self.model_prefix = model_prefix
        self.gcn = gcn
        self._load_model()

    def _load_model(self):
        self.model = load_model(self.model_prefix + '.hdf5', custom_objects={'GraphCNN': GraphCNN, 'micro_aupr': micro_aupr})
        metadata = pickle.load(open(self.model_prefix + '_metadata.pckl', 'rb'))
        #self.thresh = metadata['thresh']
        self.thresh  = 0.1 * np.ones(len(metadata['goterms']))
        self.gonames = metadata['gonames']
        self.goterms = metadata['goterms']

    def predict(self, test_prot, chain='query_prot'):
        print ("### Computing predictions on a single protein...")
        self.Y_hat = np.zeros((1, len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        if self.gcn:
            cmap = np.load(test_prot)
            A = cmap['A_ca_10A']
            S = seq2onehot(str(cmap['sequence']))
            S = S.reshape(1, *S.shape)
            A = A.reshape(1, *A.shape)
            y = self.model.predict([A, S]).reshape(-1)
            self.Y_hat[0] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[A, S], str(cmap['sequence'])]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))
        else:
            S = seq2onehot(str(test_prot))
            S = S.reshape(1, *S.shape)
            y = self.model.predict(S).reshape(-1)
            self.Y_hat[0] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[S], test_prot]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))

    def predict_from_catalogue(self, catalogue_fn):
        print ("### Computing predictions from catalogue...")
        self.chain2path = load_catalogue(catalogue_fn)
        test_prot_list = list(self.chain2path.keys())
        self.Y_hat = np.zeros((len(test_prot_list), len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        if self.gcn:
            for i, chain in enumerate(test_prot_list):
                cmap = np.load(self.chain2path[chain])
                A = cmap['A_ca_10A']
                S = seq2onehot(str(cmap['sequence']))
                S = S.reshape(1, *S.shape)
                A = A.reshape(1, *A.shape)
                y = self.model.predict([A, S]).reshape(-1)
                self.Y_hat[i] = y
                self.prot2goterms[chain] = []
                self.data[chain] = [[A, S], str(cmap['sequence'])]
                go_idx = np.where((y >= self.thresh) == True)[0]
                for idx in go_idx:
                    if idx not in self.goidx2chains:
                        self.goidx2chains[idx] = set()
                    self.goidx2chains[idx].add(chain)
                    self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))
        else:
            for i, chain in enumerate(test_prot_list):
                cmap = np.load(self.chain2path[chain])
                S = seq2onehot(str(cmap['sequence']))
                S = S.reshape(1, *S.shape)
                y = self.model.predict(S).reshape(-1)
                self.Y_hat[i] = y
                self.prot2goterms[chain] = []
                self.data[chain] = [[S], str(cmap['sequence'])]
                go_idx = np.where((y >= self.thresh) == True)[0]
                for idx in go_idx:
                    if idx not in self.goidx2chains:
                        self.goidx2chains[idx] = set()
                    self.goidx2chains[idx].add(chain)
                    self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))

    def predict_from_fasta(self, fasta_fn):
        print ("### Computing predictions from fasta...")
        test_prot_list, sequences = load_FASTA(fasta_fn)
        self.Y_hat = np.zeros((len(test_prot_list), len(self.goterms)), dtype=float)
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}

        for i, chain in enumerate(test_prot_list):
            S = seq2onehot(str(sequences[i]))
            S = S.reshape(1, *S.shape)
            y = self.model.predict(S).reshape(-1)
            self.Y_hat[i] = y
            self.prot2goterms[chain] = []
            self.data[chain] = [[S], str(sequences[i])]
            go_idx = np.where((y >= self.thresh) == True)[0]
            for idx in go_idx:
                if idx not in self.goidx2chains:
                    self.goidx2chains[idx] = set()
                self.goidx2chains[idx].add(chain)
                self.prot2goterms[chain].append((self.goterms[idx], self.gonames[idx], float(y[idx])))

    def save_predictions(self, output_fn):
        print ("### Saving predictions to *.pckl file...")
        pickle.dump({'goidx2chains': self.goidx2chains, 'Y_hat': self.Y_hat, 'goterms': self.goterms, 'gonames': self.gonames}, open(output_fn, 'wb'))

    def export_csv(self, output_fn, verbose):
        with open(output_fn, 'w') as csvFile:
            writer = csv.writer(csvFile, delimiter=',', quotechar='"')
            writer.writerow(['### Predictions made by DeepFRI.'])
            writer.writerow(['Protein', 'GO_term/EC_number', 'Score', 'GO_term/EC_number name'])
            if verbose:
                print ('Protein', 'GO-term/EC-number', 'Score', 'GO-term/EC-number name')
            for prot in self.prot2goterms:
                sorted_rows = sorted(self.prot2goterms[prot], key=lambda x: x[2], reverse=True)
                for row in sorted_rows:
                    if verbose:
                        print (prot, row[0], '{:.5f}'.format(row[2]), row[1])
                    writer.writerow([prot, row[0], '{:.5f}'.format(row[2]), row[1]])
        csvFile.close()

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
        print ("### Computing gradCAM for each function of every predicted protein...")
        self.pdb2cam = {}
        for go_indx in self.goidx2chains:
            input_data = []
            pred_chains = list(self.goidx2chains[go_indx])
            print ("### Computing gradCAM for ", self.gonames[go_indx], '... [# proteins=', len(pred_chains), ']')
            for chain in pred_chains:
                input_data.append(self.data[chain][0])
                if chain not in self.pdb2cam:
                    self.pdb2cam[chain] = {}
                    self.pdb2cam[chain]['GO_ids'] = []
                    self.pdb2cam[chain]['GO_names'] = []
                    self.pdb2cam[chain]['sequence'] = None
                    self.pdb2cam[chain]['saliency_maps'] = []
                self.pdb2cam[chain]['GO_ids'].append(self.goterms[go_indx])
                self.pdb2cam[chain]['GO_names'].append(self.gonames[go_indx])
                self.pdb2cam[chain]['sequence'] = self.data[chain][1]
            heatmaps = self._gradCAM(input_data, go_indx, layer_name=layer_name)
            for i, chain in enumerate(pred_chains):
                self.pdb2cam[chain]['saliency_maps'].append(heatmaps[i])

    def save_gradCAM(self, output_fn):
        print ("### Saving CAMs to *.pckl file...")
        pickle.dump(self.pdb2cam, open(output_fn, 'wb'))
