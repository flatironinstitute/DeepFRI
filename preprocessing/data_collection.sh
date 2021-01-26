#!/bin/bash

DATA_DIR=./data/
TFR_DIR=/mnt/ceph/users/vgligorijevic/ContactMaps/TFRecords/
SEQ_SIM=95

mkdir $DATA_DIR
printf "\n\n  DATA DIRECTORY (%s) CREATED!\n" $DATA_DIR

printf "\n\n  DOWNLOADING SIFTS-GO DATA...\n"
wget ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_go.tsv.gz -O $DATA_DIR/pdb_chain_go.tsv.gz

printf "\n\n  DOWNLOADING SIFTS-EC DATA...\n"
wget ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_enzyme.tsv.gz -O $DATA_DIR/pdb_chain_enzyme.tsv.gz

printf "\n\n  DOWNLOADING PDB SEQRES SEQUENCES...\n"
wget ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz -O $DATA_DIR/pdb_seqres.txt.gz

printf "\n\n  DOWNLOADING PDB CLUSTERS...\n"
wget https://cdn.rcsb.org/resources/sequence/clusters/bc-$SEQ_SIM.out -O $DATA_DIR/bc-$SEQ_SIM.out

printf "\n\n  DOWNLOADING GO HIERARCHY...\n"
wget http://purl.obolibrary.org/obo/go/go-basic.obo -O $DATA_DIR/go-basic.obo

printf "\n\n  PREPROCESSING GO-ANNOTATIONS [Please wait this process may take a few minutes]...\n"
python create_nrPDB_GO_annot.py \
    -sifts $DATA_DIR/pdb_chain_go.tsv.gz \
    -bc $DATA_DIR/bc-$SEQ_SIM.out \
    -seqres $DATA_DIR/pdb_seqres.txt.gz \
    -obo $DATA_DIR/go-basic.obo \
    -out $DATA_DIR/nrPDB-GO \

printf "\n\n  PREPROCESSING EC-ANNOTATIONS [Please wait this process may take a few minutes]...\n"
python create_nrPDB_EC_annot.py \
    -sifts $DATA_DIR/pdb_chain_enzyme.tsv.gz \
    -bc $DATA_DIR/bc-$SEQ_SIM.out \
    -seqres $DATA_DIR/pdb_seqres.txt.gz \
    -out $DATA_DIR/nrPDB-EC \

printf "\n\n  RETRIEVING PDB FILES AND CREATING DISTANCE MAPS...\n"
mkdir $DATA_DIR/annot_pdb_chains_npz/
python PDB2distMap.py \
    -annot $DATA_DIR/nrPDB-GO_annot.tsv \
    -seqres $DATA_DIR/pdb_seqres.txt.gz \
    -num_threads 20 \
    -bc $DATA_DIR/bc-$SEQ_SIM.out \
    -out_dir $DATA_DIR/annot_pdb_chains_npz/ \

python PDB2distMap.py \
    -annot $DATA_DIR/nrPDB-EC_annot.tsv \
    -ec \
    -seqres $DATA_DIR/pdb_seqres.txt.gz \
    -num_threads 20 \
    -bc $DATA_DIR/bc-$SEQ_SIM.out \
    -out_dir $DATA_DIR/annot_pdb_chains_npz/ \

rm -r obsolete/

printf "\n\n  CREATE TFRecord FILES..."
mkdir $TFR_DIR
python PDB2TFRecord.py \
    -annot $DATA_DIR/nrPDB-GO_annot.tsv \
    -prot_list $DATA_DIR/nrPDB-GO_train.txt \
    -npz_dir $DATA_DIR/annot_pdb_chains_npz/ \
    -num_shards 30 \
    -num_threads 30 \
    -tfr_prefix $TFR_DIR/PDB_GO_train \

python PDB2TFRecord.py \
    -annot $DATA_DIR/nrPDB-GO_annot.tsv \
    -prot_list $DATA_DIR/nrPDB-GO_valid.txt \
    -npz_dir $DATA_DIR/annot_pdb_chains_npz/ \
    -num_shards 3 \
    -num_threads 3 \
    -tfr_prefix $TFR_DIR/PDB_GO_valid \

python PDB2TFRecord.py \
    -annot $DATA_DIR/nrPDB-EC_annot.tsv \
    -ec \
    -prot_list $DATA_DIR/nrPDB-EC_train.txt \
    -npz_dir $DATA_DIR/annot_pdb_chains_npz/ \
    -num_shards 15 \
    -num_threads 15 \
    -tfr_prefix $TFR_DIR/PDB_EC_train \

python PDB2TFRecord.py \
    -annot $DATA_DIR/nrPDB-EC_annot.tsv \
    -ec \
    -prot_list $DATA_DIR/nrPDB-EC_valid.txt \
    -npz_dir $DATA_DIR/annot_pdb_chains_npz/ \
    -num_shards 1 \
    -num_threads 1 \
    -tfr_prefix $TFR_DIR/PDB_EC_valid \
