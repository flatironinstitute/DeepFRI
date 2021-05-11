#!/usr/bin/env python

from create_nrPDB_GO_annot import read_fasta, load_clusters
from biotoolbox.structure_file_reader import build_structure_container_for_pdb
from biotoolbox.contact_map_builder import DistanceMapBuilder
from Bio.PDB import PDBList

from functools import partial
import numpy as np
import argparse
import csv
import os


def make_distance_maps(pdbfile, chain=None, sequence=None):
    """
    Generate (diagonalized) C_alpha and C_beta distance matrix from a pdbfile
    """
    pdb_handle = open(pdbfile, 'r')
    structure_container = build_structure_container_for_pdb(pdb_handle.read(), chain).with_seqres(sequence)
    # structure_container.chains = {chain: structure_container.chains[chain]}

    mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)  # start with CA distances
    ca = mapper.generate_map_for_pdb(structure_container)
    cb = mapper.set_atom("CB").generate_map_for_pdb(structure_container)
    pdb_handle.close()

    return ca.chains, cb.chains


def load_GO_annot(filename):
    """ Load GO annotations """
    onts = ['molecular_function', 'biological_process', 'cellular_component']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                prot2annot[prot][onts[i]] = [goterm for goterm in prot_goterms[i].split(',') if goterm != '']
    return prot2annot, goterms, gonames


def load_EC_annot(filename):
    """ Load EC annotations """
    prot2annot = {}
    ec_numbers = []
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        ec_numbers = next(reader)
        next(reader, None)  # skip the headers
        for row in reader:
            prot, prot_ec_numbers = row[0], row[1]
            prot2annot[prot] = [ec_num for ec_num in prot_ec_numbers.split(',')]
    return prot2annot, ec_numbers


def retrieve_pdb(pdb, chain, chain_seqres, pdir):
    pdb_list = PDBList()
    pdb_list.retrieve_pdb_file(pdb, pdir=pdir)
    ca, cb = make_distance_maps(pdir + '/' + pdb +'.cif', chain=chain, sequence=chain_seqres)

    return ca[chain]['contact-map'], cb[chain]['contact-map']


def load_list(fname):
    """
    Load PDB chains
    """
    pdb_chain_list = set()
    fRead = open(fname, 'r')
    for line in fRead:
        pdb_chain_list.add(line.strip())
    fRead.close()

    return pdb_chain_list


def write_annot_npz(prot, prot2seq=None, out_dir=None):
    """
    Write to *.npz file format.
    """
    pdb, chain = prot.split('-')
    print ('pdb=', pdb, 'chain=', chain)
    try:
        A_ca, A_cb = retrieve_pdb(pdb.lower(), chain, prot2seq[prot], pdir=os.path.join(out_dir, 'tmp_PDB_files_dir'))
        np.savez_compressed(os.path.join(out_dir, prot),
                            C_alpha=A_ca,
                            C_beta=A_cb,
                            seqres=prot2seq[prot],
                            )
    except Exception as e:
        print (e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-annot', type=str, help="Input file (*.tsv) with preprocessed annotations.")
    parser.add_argument('-ec', help="Use EC annotations.", action="store_true")
    parser.add_argument('-seqres', type=str, default='./data/pdb_seqres.txt.gz', help="PDB chain seqres fasta.")
    parser.add_argument('-num_threads', type=int, default=20, help="Number of threads (CPUs) to use in the computation.")
    parser.add_argument('-bc', type=str, help="Clusters of PDB chains computd by Blastclust.")
    parser.add_argument('-out_dir', type=str, default='./data/annot_pdb_chains_npz/', help="Output directory with distance maps saved in *.npz format.")
    args = parser.parse_args()

    # load annotations
    prot2goterms = {}
    if args.annot is not None:
        if args.ec:
            prot2goterms, _ = load_EC_annot(args.annot)
        else:
            prot2goterms, _, _ = load_GO_annot(args.annot)
        print ("### number of annotated proteins: %d" % (len(prot2goterms)))

    # load sequences
    prot2seq = read_fasta(args.seqres)
    print ("### number of proteins with seqres sequences: %d" % (len(prot2seq)))

    # load clusters
    pdb2clust = {}
    if args.bc is not None:
        pdb2clust = load_clusters(args.bc)
        clusters = set([pdb2clust[prot][0] for prot in prot2goterms])
        print ("### number of annotated clusters: %d" % (len(clusters)))

    """
    # extracting unannotated proteins
    unannot_prots = set()
    for prot in pdb2clust:
        if (pdb2clust[prot][0] not in clusters) and (pdb2clust[prot][1] == 0) and (prot in prot2seq):
            unannot_prots.add(prot)
    print ("### number of unannot proteins: %d" % (len(unannot_prots)))
    """

    to_be_processed = set(prot2seq.keys())
    if len(prot2goterms) != 0:
        to_be_processed = to_be_processed.intersection(set(prot2goterms.keys()))
    if len(prot2goterms) != 0:
        to_be_processed = to_be_processed.intersection(set(pdb2clust.keys()))
    print ("Number of pdbs to be processed=", len(to_be_processed))
    print (to_be_processed)

    # process on multiple cpus
    nprocs = args.num_threads
    out_dir = args.out_dir
    import multiprocessing
    nprocs = np.minimum(nprocs, multiprocessing.cpu_count())
    if nprocs > 4:
        pool = multiprocessing.Pool(processes=nprocs)
        pool.map(partial(write_annot_npz, prot2seq=prot2seq, out_dir=out_dir),
                 to_be_processed)
    else:
        for prot in to_be_processed:
            write_annot_npz(prot, prot2seq=prot2seq, out_dir=out_dir)
