from functools import partial
import multiprocessing

import networkx as nx
import numpy as np
import argparse
import secrets

import glob
import os


def write_orca_input_file(G, fname):
    """Write graph edgelist file."""
    fWrite = open(fname, 'w')
    fWrite.write("%d %d\n" % (G.order(), G.size()))
    for (n1, n2) in G.edges():
        fWrite.write("%d %d\n" % (n1, n2))
    fWrite.close()


def read_orca_output_file(fname):
    """Read graphlet counts file."""
    A = np.loadtxt(fname, delimiter=" ")

    return A


def compute_graphlet_degree_vectors(cmap_npz_file, out_dir, cmap_key='C_alpha', thresh=8.0):
    cmap = np.load(cmap_npz_file)
    if cmap_key not in cmap:
        print ("### Wrong dict key for cmap.")
    D = cmap[cmap_key]

    # construct contact map
    A = np.double(D <= thresh)
    A = A - np.diag(np.diag(A))

    # create graph from adj
    G = nx.from_numpy_matrix(A)

    # PDB ID:
    pdb_id = cmap_npz_file.split('/')[-1].split('.')[0]

    # run orca.exe to obtain graphlet vector for each node
    tmp = "".join([secrets.token_hex(10), pdb_id])
    write_orca_input_file(G, fname="".join([tmp, ".in"]))
    command = "".join(['./orca.exe 5 ', tmp + '.in', ' ', tmp + '.out'])
    os.system(command)

    # load graphlet degree vectors
    node_GDV = read_orca_output_file("".join([tmp, ".out"]))
    np.savez_compressed(os.path.join(out_dir, pdb_id),
                        node_GDV=node_GDV,
                        C_alpha=cmap['C_alpha'],
                        C_beta=cmap['C_beta'],
                        seqres=cmap['seqres'])

    # remove tmp file
    os.system('rm ' + tmp + '*')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_dir', type=str, default='./input_dir/',
                        help="Input directory with distance maps saved in *.npz format to be loaded.")
    parser.add_argument('-o', '--output_dir', type=str, default='./output_dir/',
                        help="Output directory with graphlet degree matrix saved in *.npz format.")
    parser.add_argument('--num_threads', type=int, default=20, help="Number of CPUs to use in the computation.")
    parser.add_argument('--cmap_key', type=str, default='C_alpha', help="Dict key for extracting distance map from npz file.")
    parser.add_argument('--thresh', type=float, default=8.0, help="Distance threshold for constrcting contact maps.")
    args = parser.parse_args()


    out_dir = args.output_dir
    cmap_key = args.cmap_key
    thresh = args.thresh

    pool = multiprocessing.Pool(processes=args.num_threads)
    func = partial(compute_graphlet_degree_vectors, out_dir=out_dir, cmap_key=cmap_key, thresh=thresh)
    pool.map(func, glob.glob(args.input_dir + '/*.npz'))
