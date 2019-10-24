import pickle
import argparse
import numpy as np

import keras.backend as K
# from keras.models import load_model
# from keras_gcnn.layers import GraphCNN
# from utils import micro_aupr,
from utils import load_catalogue, rnd_adj

from deepfrier import DeepFRI
from utils import get_thresholds, seq2onehot


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hidden_dims', type=int, default=[512], nargs='+', help="Dimensions of hidden layers.")
    parser.add_argument('--gcn_dims', type=int, default=[128, 128, 256], nargs='+', help="Dimensions of GCN layers.")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate.")
    parser.add_argument('--l2_reg', type=float, default=1e-3, help="L2 regularization coefficient.")
    parser.add_argument('--lr', type=float, default=0.0002, help="Initial learning rate.")
    parser.add_argument('--epochs', type=int, default=200, help="Number of epochs to train.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--pad_len', type=int, default=1200, help="Padd length.")
    parser.add_argument('--results_dir', type=str, default='./results/', help="Path to directory with results and models.")
    parser.add_argument('--ont', type=str, default='molecular_function', help="Ontology.")
    parser.add_argument('--cmap_type', type=str, default='A_ca', help="Contact maps type.")
    parser.add_argument('--model_name', type=str, default='GCN_MF', help="Name of the GCN model.")
    parser.add_argument('--lm_model_name', type=str, help="Name of the LM model.")
    parser.add_argument('--split_fn', type=str,
                        default='/mnt/home/vgligorijevic/Projects/NewMethods/Contact_maps/go_annot/pdb_GO_train_test_split_bc_30.pckl',
                        help="Train-Test split for model's trainig.")
    parser.add_argument('--catalogue', type=str, default="/mnt/ceph/users/vgligorijevic/ContactMaps/data/nr_pdb_chains/catalogue.csv",
                        help="Catalogue with chain to file (in *.npz format) path mapping.")
    parser.add_argument('--train_tfrecord_fn', type=str,
                        default="/mnt/ceph/users/vgligorijevic/ContactMaps/data/TFRecord/pdb_chains_seqid_30_molecular_function_train_EXP.tfrecords",
                        help="Train tfrecords.")
    parser.add_argument('--valid_tfrecord_fn', type=str,
                        default="/mnt/ceph/users/vgligorijevic/ContactMaps/data/TFRecord/pdb_chains_seqid_30_molecular_function_valid_EXP.tfrecords",
                        help="Valid tfrecords.")

    args = parser.parse_args()
    print (args)

    annot = pickle.load(open(args.split_fn, 'rb'))
    goterms = annot[args.ont]['goterms']
    gonames = annot[args.ont]['gonames']
    Y_train = np.concatenate([annot[args.ont]['Y_train'], annot[args.ont]['Y_valid']], axis=0)
    Y_test = annot[args.ont]['Y_test']
    Y_valid2 = annot[args.ont]['Y_valid2']
    test_chains = annot[args.ont]['test_pdb_chains']
    valid2_chains = annot[args.ont]['valid2_pdb_chains']

    if args.train_tfrecord_fn.find('IEA') > 0:
        Y_train[Y_train == -1] = 1
        Y_valid2[Y_valid2 == -1] = 1
    else:
        Y_train[Y_train == -1] = 0
        Y_valid2[Y_valid2 == -1] = 0

    K.get_session()
    print ("### Training model: ", args.model_name, " on ", len(goterms), " GO terms.")
    model = DeepFRI(output_dim=len(goterms), results_dir=args.results_dir, n_channels=26, gcn_dims=args.gcn_dims,
                    hidd_dims=args.hidden_dims, lr=args.lr, drop=args.dropout, l2_reg=args.l2_reg, lm_model_name=args.lm_model_name)
    model.train(args.train_tfrecord_fn, args.valid_tfrecord_fn, args.model_name, epochs=args.epochs,
                batch_size=args.batch_size, pad_len=args.pad_len, cmap_type=args.cmap_type)

    # model = load_model(args.results_dir + args.model_name + '.h5', custom_objects={'GraphCNN': GraphCNN, 'micro_aupr': micro_aupr})
    # print (model.summary())

    # compute thresholds on valid2 chains
    print ("### Computing thresholds...")
    chain2path = load_catalogue(fn=args.catalogue)
    Y_hat_valid2 = np.zeros_like(Y_valid2)
    for i, chain in enumerate(valid2_chains):
        cmap = np.load(chain2path[chain])
        A = cmap['A_ca_10A']
        S = seq2onehot(str(cmap['sequence']))
        # ##
        S = S.reshape(1, *S.shape)
        A = A.reshape(1, *A.shape)
        Y_hat_valid2[i] = model.predict([A, S])
    thresholds = get_thresholds(Y_valid2, Y_hat_valid2)
    pickle.dump({'thresh': thresholds, 'goterms': goterms, 'gonames': gonames}, open(args.results_dir + args.model_name + '_thresholds.pckl', 'wb'))

    # compute perf on test chains
    print ("### Computing predictions on test set...")
    Y_hat_test = np.zeros_like(Y_test, dtype=float)
    Y_dummy_test = np.zeros_like(Y_test, dtype=float)
    Y_rnd_test = np.zeros_like(Y_test, dtype=float)
    lengths = []
    for i, chain in enumerate(test_chains):
        cmap = np.load(chain2path[chain])
        A = cmap['A_ca_10A']
        S = seq2onehot(str(cmap['sequence']))
        L = cmap['L'].item()
        p = (np.count_nonzero(A) - L)/float(L*(L-1))

        I = np.eye(L)
        A_rnd = rnd_adj(L, p)
        # ##
        S = S.reshape(1, *S.shape)
        A = A.reshape(1, *A.shape)
        A_rnd = A_rnd.reshape(1, *A_rnd.shape)
        I = I.reshape(1, *I.shape)
        # ##
        Y_hat_test[i] = model.predict([A, S])
        Y_dummy_test[i] = model.predict([I, S])
        Y_rnd_test[i] = model.predict([A_rnd, S])
        lengths.append(L)
    lengths = np.asarray(lengths)
    num_pos = Y_train.sum(axis=0)
    pickle.dump({'Y_test': Y_test, 'Y_hat_test': Y_hat_test, 'num_pos': num_pos, 'L_test': lengths,
                 'Y_dummy_test': Y_dummy_test, 'Y_rnd_test': Y_rnd_test, 'goterms': goterms, 'gonames': gonames},
                open(args.results_dir + args.model_name + '_pred_scores.pckl', 'wb'))

    # making predictions on PDB chains
    if args.train_tfrecord_fn.find('swiss-model') >= 0:
        annot = pickle.load(open('/mnt/home/vgligorijevic/Projects/NewMethods/Contact_maps/go_annot/pdb_GO_train_test_split_bc_30.pckl', 'rb'))
        test_goterms = annot[args.ont]['goterms']
        test_gonames = annot[args.ont]['gonames']
        Y_test = annot[args.ont]['Y_test']
        chains = annot[args.ont]['test_pdb_chains']

        chain2path = load_catalogue(fn='/mnt/ceph/users/vgligorijevic/ContactMaps/data/nr_pdb_chains/catalogue.csv')
        Y_hat_test = np.zeros((len(chains), len(goterms)), dtype=float)
        for i, chain in enumerate(chains):
            cmap = np.load(chain2path[chain])
            A = cmap['A_ca_10A']
            S = seq2onehot(str(cmap['sequence']))
            # ##
            S = S.reshape(1, *S.shape)
            A = A.reshape(1, *A.shape)
            Y_hat_test[i] = model.predict([A, S])

        overlap_goterms = list(set(goterms).intersection(set(test_goterms)))
        print ("### Making predictions for: ", len(goterms), "goterms.")

        train_idx = [np.where(goterms == go)[0][0] for go in overlap_goterms]
        test_idx = [np.where(test_goterms == go)[0][0] for go in overlap_goterms]

        pickle.dump({'Y_test': Y_test[:, test_idx], 'Y_hat_test': Y_hat_test[:, train_idx], 'chain_ids': chains,
                    'goterms': test_goterms[test_idx], 'gonames': test_gonames[test_idx]},
                    open(args.results_dir + args.model_name + '_test_seqid_30_pred_scores.pckl', 'wb'))
    # save models
    model.plot_losses()
    model.save_model()
    K.clear_session()
