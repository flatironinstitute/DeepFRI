from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_protein
from Bio.SeqRecord import SeqRecord

import numpy as np
import argparse
import gzip
import csv


# ## packages (dependencies):
# csv, biopython

# ## input data:
# pdb_chain_enzyme.tsv (from: https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html)
# bc-95.out (from: ftp://resources.rcsb.org/sequence/clusters/)


def read_fasta(fn_fasta):
    aa = set(['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E'])
    prot2seq = {}
    with gzip.open(fn_fasta, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            prot = record.id
            pdb, chain = prot.split('_')
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


def nr_set(chains, pdb2seq, pdb2clust):
    clust2chain = {}
    for chain in chains:
        if chain in pdb2seq and chain in pdb2clust:
            c_idx = pdb2clust[chain][0]
            if c_idx not in clust2chain:
                clust2chain[c_idx] = chain
            else:
                _chain = clust2chain[c_idx]
                if pdb2clust[chain][1] < pdb2clust[_chain][1]:
                    clust2chain[c_idx] = chain
    return set(clust2chain.values())


def read_sifts(fname, chains):
    print ("### Loading SIFTS annotations...")
    pdb2ec = {}
    ec2info = {}
    with gzip.open(fname, mode='rt') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)  # skip the headers
        next(reader, None)  # skip the headers
        for row in reader:
            pdb = row[0].strip().upper()
            chain = row[1].strip()
            ec_number = row[3].strip()
            level = 4 - len(np.where(np.asarray(ec_number.split('.')) == '-')[0])
            pdb_chain = pdb + '-' + chain
            if pdb_chain in chains and level > 2:
                if pdb_chain not in pdb2ec:
                    pdb2ec[pdb_chain] = set()
                # EC numbers
                ec_numbers = []
                if level == 4:
                    ec_numbers.append(ec_number)
                    ec_numbers.append('.'.join(ec_number.split('.')[:3] + ['-']))
                else:
                    ec_numbers.append(ec_number)
                for ec_num in ec_numbers:
                    pdb2ec[pdb_chain].add(ec_num)
                    if ec_num not in ec2info:
                        ec2info[ec_num] = set()
                    ec2info[ec_num].add(pdb_chain)

    return pdb2ec, ec2info


def write_output_files(fname, pdb2ec, ec2info, pdb2seq, thresh=10):
    # select EC numbers (> thresh)
    selected_ec_numbers = set()
    selected_proteins = set()
    for ec_number in ec2info:
        prots = ec2info[ec_number]
        num = len(prots)
        if num > thresh:
            selected_ec_numbers.add(ec_number)
            selected_proteins = selected_proteins.union(prots)

    print ("### Total Proteins: %d" % (len(selected_proteins)))
    print ("### Total EC: %d" % (len(selected_ec_numbers)))

    level_3_ec = 0
    level_4_ec = 0
    for ec_num in selected_ec_numbers:
        if ec_num.find('-') == -1:
            level_4_ec += 1
        else:
            level_3_ec += 1

    print ("### 3rd level EC: %d" % (level_3_ec))
    print ("### 4th level EC: %d" % (level_4_ec))

    sequences_list = []
    protein_list = []
    with open(fname + '_annot.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["### EC-numbers"])
        tsv_writer.writerow(selected_ec_numbers)
        tsv_writer.writerow(["### PDB-chain", "EC-numbers"])
        for chain in selected_proteins:
            ec_numbers = set(pdb2ec[chain])
            ec_numbers = ec_numbers.intersection(selected_ec_numbers)
            if len(ec_numbers) > 0 and chain in pdb2seq:
                sequences_list.append(SeqRecord(Seq(pdb2seq[chain], generic_protein), id=chain, description="nrPDB"))
                protein_list.append(chain)
                tsv_writer.writerow([chain, ','.join(ec_numbers)])

    np.random.seed(1234)
    np.random.shuffle(protein_list)
    print ("Total number of annot nrPDB=%d" % (len(protein_list)))

    test_list = set()
    test_ec_coverage = set()
    test_sequences_list = []
    i = 0
    while len(test_list) < int(0.1*len(protein_list)):
        ec_numbers = pdb2ec[protein_list[i]]
        ec_numbers = ec_numbers.intersection(selected_ec_numbers)
        if len(ec_numbers) > 1:
            test_ec_coverage = test_ec_coverage.union(ec_numbers)
            test_list.add(protein_list[i])
            test_sequences_list.append(SeqRecord(Seq(pdb2seq[protein_list[i]], generic_protein), id=protein_list[i], description="nrPDB_test"))
        i += 1

    print ("Total number of test nrPDB=%d" % (len(test_list)))
    print ("Test EC coverage=%d" % (len(test_ec_coverage)))

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
    parser.add_argument('-sifts', type=str, default='./data/pdb_chain_enzyme.tsv.gz', help="SIFTS annotation files.")
    parser.add_argument('-bc', type=str, default='./data/bc-95.out', help="Blastclust of PDB chains.")
    parser.add_argument('-seqres', type=str, default='./data/pdb_seqres.txt.gz', help="PDB chain seqres fasta.")
    parser.add_argument('-out', type=str, default='./data/nrPDB-EC_2020.04', help="Output filename prefix.")
    args = parser.parse_args()

    annoted_chains = load_pdbs(args.sifts)
    pdb2clust = load_clusters(args.bc)
    pdb2seq = read_fasta(args.seqres)
    nr_chains = nr_set(annoted_chains, pdb2seq, pdb2clust)
    print ("### nrPDB annotated chains=", len(nr_chains))

    pdb2go, go2info = read_sifts(args.sifts, nr_chains)
    print ("### chains=%d; ec_numbers=%d" % (len(pdb2go), len(go2info)))
    write_output_files(args.out, pdb2go, go2info, pdb2seq)
