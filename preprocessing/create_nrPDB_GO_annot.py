from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_protein
from Bio.SeqRecord import SeqRecord

import networkx as nx
import numpy as np
import argparse
import obonet
import gzip
import csv


# ## packages (dependencies):
# networkx, obonet, csv, biopython

# ## input data:
# go-basic.obo (from: http://geneontology.org/docs/download-ontology/)
# pdb_chain_go.tsv (from: https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html)
# bc-95.out (from: ftp://resources.rcsb.org/sequence/clusters/)

exp_evidence_codes = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC', 'CURATED'])
root_terms = set(['GO:0008150', 'GO:0003674', 'GO:0005575'])


def read_fasta(fn_fasta):
    aa = set(['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E'])
    prot2seq = {}
    if fn_fasta.endswith('gz'):
        handle = gzip.open(fn_fasta, "rt")
    else:
        handle = open(fn_fasta, "rt")

    for record in SeqIO.parse(handle, "fasta"):
        seq = str(record.seq)
        prot = record.id
        pdb, chain = prot.split('_') if '_' in prot else prot.split('-')
        prot = pdb.upper() + '-' + chain
        if len(seq) >= 60 and len(seq) <= 1000:
            if len((set(seq).difference(aa))) == 0:
                prot2seq[prot] = seq

    return prot2seq


def write_prot_list(protein_list, filename):
    # write list of protein IDs
    fWrite = open(filename, 'w')
    for p in protein_list:
        fWrite.write("%s\n" % (p))
    fWrite.close()


def write_fasta(fn, sequences):
    # write fasta
    with open(fn, "w") as output_handle:
        for sequence in sequences:
            SeqIO.write(sequence, output_handle, "fasta")


def load_go_graph(fname):
    # read *.obo file
    go_graph = obonet.read_obo(open(fname, 'r'))
    return go_graph


def load_pdbs(sifts_fname):
    # read annotated PDB chains
    pdb_chains = set()
    with gzip.open(sifts_fname, mode='rt') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)  # skip the headers
        next(reader, None)  # skip the headers
        for row in reader:
            pdb = row[0].strip().upper()
            chain = row[1].strip()
            pdb_chains.add(pdb + '-' + chain)
    return pdb_chains


def load_clusters(fname):
    pdb2clust = {}  # (c_idx, rank)
    c_ind = 1
    fRead = open(fname, 'r')
    for line in fRead:
        clust = line.strip().split()
        clust = [p.replace('_', '-') for p in clust]
        for rank, p in enumerate(clust):
            pdb2clust[p] = (c_ind, rank)
        c_ind += 1
    fRead.close()
    return pdb2clust


def nr_set(chains, pdb2clust):
    clust2chain = {}
    for chain in chains:
        if chain in pdb2clust:
            c_idx = pdb2clust[chain][0]
            if c_idx not in clust2chain:
                clust2chain[c_idx] = chain
            else:
                _chain = clust2chain[c_idx]
                if pdb2clust[chain][1] < pdb2clust[_chain][1]:
                    clust2chain[c_idx] = chain
    return set(clust2chain.values())


def read_sifts(fname, chains, go_graph):
    print ("### Loading SIFTS annotations...")
    pdb2go = {}
    go2info = {}
    with gzip.open(fname, mode='rt') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)  # skip the headers
        next(reader, None)  # skip the headers
        for row in reader:
            pdb = row[0].strip().upper()
            chain = row[1].strip()
            evidence = row[4].strip()
            go_id = row[5].strip()
            pdb_chain = pdb + '-' + chain
            if (pdb_chain in chains) and (go_id in go_graph) and (go_id not in root_terms):
                if pdb_chain not in pdb2go:
                    pdb2go[pdb_chain] = {'goterms': [go_id], 'evidence': [evidence]}
                namespace = go_graph.nodes[go_id]['namespace']
                go_ids = nx.descendants(go_graph, go_id)
                go_ids.add(go_id)
                go_ids = go_ids.difference(root_terms)
                for go in go_ids:
                    pdb2go[pdb_chain]['goterms'].append(go)
                    pdb2go[pdb_chain]['evidence'].append(evidence)
                    name = go_graph.nodes[go]['name']
                    if go not in go2info:
                        go2info[go] = {'ont': namespace, 'goname': name, 'pdb_chains': set([pdb_chain])}
                    else:
                        go2info[go]['pdb_chains'].add(pdb_chain)
    return pdb2go, go2info


def write_output_files(fname, pdb2go, go2info, pdb2seq):
    # select goterms (> 49, < 5000)
    onts = ['molecular_function', 'biological_process', 'cellular_component']
    selected_goterms = {ont: set() for ont in onts}
    selected_proteins = set()
    for goterm in go2info:
        prots = go2info[goterm]['pdb_chains']
        num = len(prots)
        namespace = go2info[goterm]['ont']
        if num > 49 and num <= 5000:
            selected_goterms[namespace].add(goterm)
            selected_proteins = selected_proteins.union(prots)
    """
    for chain in pdb2go:
        goterms = set(pdb2go[chain]['goterms'])
        if len(goterms) > 2 and chain in pdb2seq:
            for goterm in goterms:
                prots = go2info[goterm]['pdb_chains']
                num = len(prots)
                namespace = go2info[goterm]['ont']
                if num > 19 and num <= 5000:
                    selected_goterms[namespace].add(goterm)
                    selected_proteins = selected_proteins.union(prots)
    """

    selected_goterms_list = {ont: list(selected_goterms[ont]) for ont in onts}
    selected_gonames_list = {ont: [go2info[goterm]['goname'] for goterm in selected_goterms_list[ont]] for ont in onts}

    for ont in onts:
        print ("###", ont, ":", len(selected_goterms_list[ont]))

    sequences_list = []
    protein_list = []
    with open(fname + '_annot.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for ont in onts:
            tsv_writer.writerow(["### GO-terms (%s)" % (ont)])
            tsv_writer.writerow(selected_goterms_list[ont])
            tsv_writer.writerow(["### GO-names (%s)" % (ont)])
            tsv_writer.writerow(selected_gonames_list[ont])
        tsv_writer.writerow(["### PDB-chain", "GO-terms (molecular_function)", "GO-terms (biological_process)", "GO-terms (cellular_component)"])
        for chain in selected_proteins:
            goterms = set(pdb2go[chain]['goterms'])
            if len(goterms) > 2:
                # selected goterms
                mf_goterms = goterms.intersection(set(selected_goterms_list[onts[0]]))
                bp_goterms = goterms.intersection(set(selected_goterms_list[onts[1]]))
                cc_goterms = goterms.intersection(set(selected_goterms_list[onts[2]]))
                if len(mf_goterms) > 0 or len(bp_goterms) > 0 or len(cc_goterms) > 0:
                    sequences_list.append(SeqRecord(Seq(pdb2seq[chain], generic_protein), id=chain, description="nrPDB"))
                    protein_list.append(chain)
                    tsv_writer.writerow([chain, ','.join(mf_goterms), ','.join(bp_goterms), ','.join(cc_goterms)])

    np.random.seed(1234)
    np.random.shuffle(protein_list)
    print ("Total number of annot nrPDB=%d" % (len(protein_list)))

    # select test set based in 30% sequence identity
    test_list = set()
    test_sequences_list = []
    i = 0
    while len(test_list) < 5000 and i < len(protein_list):
        goterms = pdb2go[protein_list[i]]['goterms']
        evidence = pdb2go[protein_list[i]]['evidence']
        goterm2evidence = {goterms[i]: evidence[i] for i in range(len(goterms))}

        # selected goterms
        mf_goterms = set(goterms).intersection(set(selected_goterms_list[onts[0]]))
        bp_goterms = set(goterms).intersection(set(selected_goterms_list[onts[1]]))
        cc_goterms = set(goterms).intersection(set(selected_goterms_list[onts[2]]))

        mf_evidence = [goterm2evidence[goterm] for goterm in mf_goterms]
        mf_evidence = [1 if evid in exp_evidence_codes else 0 for evid in mf_evidence]

        bp_evidence = [goterm2evidence[goterm] for goterm in bp_goterms]
        bp_evidence = [1 if evid in exp_evidence_codes else 0 for evid in bp_evidence]

        cc_evidence = [goterm2evidence[goterm] for goterm in cc_goterms]
        cc_evidence = [1 if evid in exp_evidence_codes else 0 for evid in cc_evidence]

        if len(mf_goterms) > 0 and len(bp_goterms) > 0 and len(cc_goterms) > 0:
            if sum(mf_evidence) > 0 and sum(bp_evidence) > 0 and sum(cc_evidence) > 0:
                test_list.add(protein_list[i])
                test_sequences_list.append(SeqRecord(Seq(pdb2seq[protein_list[i]], generic_protein), id=protein_list[i], description="nrPDB_test"))
        i += 1

    print ("Total number of test nrPDB=%d" % (len(test_list)))

    protein_list = list(set(protein_list).difference(test_list))
    np.random.shuffle(protein_list)

    idx = int(0.9*len(protein_list))
    write_prot_list(test_list, fname + '_test.txt')
    write_prot_list(protein_list[:idx], fname + '_train.txt')
    write_prot_list(protein_list[idx:], fname + '_valid.txt')
    write_fasta(fname + '_sequences.fasta', sequences_list)
    write_fasta(fname + '_test_sequences.fasta', test_sequences_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sifts', type=str, default='./data/pdb_chain_go_2019.06.18.tsv.gz', help="SIFTS annotation files.")
    parser.add_argument('-bc', type=str, default='./data/bc-95.out', help="Blastclust of PDB chains.")
    parser.add_argument('-seqres', type=str, default='./data/pdb_seqres.txt.gz', help="PDB chain seqres fasta.")
    parser.add_argument('-obo', type=str, default='./data/go-basic.obo', help="Gene Ontology hierarchy.")
    parser.add_argument('-out', type=str, default='./data/nrPDB-GO_2019.06.18', help="Output filename prefix.")
    args = parser.parse_args()

    annoted_chains = load_pdbs(args.sifts)
    pdb2clust = load_clusters(args.bc)
    pdb2seq = read_fasta(args.seqres)
    nr_chains = nr_set(annoted_chains, pdb2clust)
    print ("### nrPDB annotated chains=", len(nr_chains))

    go_graph = load_go_graph(args.obo)
    pdb2go, go2info = read_sifts(args.sifts, nr_chains, go_graph)

    write_output_files(args.out, pdb2go, go2info, pdb2seq)
