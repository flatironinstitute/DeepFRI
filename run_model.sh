# !/bin/bash
# Run GCN model

python train_DeepFRI.py \
    --gcn_dims 256 256 512 \
    --hidden_dims 1024 \
    --l2_reg 5e-4 \
    --lr 0.0002 \
    --pad_len 1200 \
    --epochs 2 \
    --cmap_type A_ca \
    --lm_model_name trained_models/lstm_lm.hdf5 \
    --ont molecular_function \
    --model_name GCN-LM_MERGED_molecular_function_EXP-IEA_seqid_30_gcn_256-256-512_hidd_1024_softmax_test \
    --catalogue /mnt/ceph/users/vgligorijevic/ContactMaps/data/Swiss-Model/merged_annot/catalogue.csv \
    --split_fn /mnt/ceph/users/vgligorijevic/ContactMaps/data/Swiss-Model/merged_annot/merged_GO_train_test_split_nr30.pckl \
    --train_tfrecord_fn /mnt/ceph/users/vgligorijevic/ContactMaps/data/Swiss-Model/merged_annot/tfrecords/*_chains_molecular_function_seqid_30_train_EXP-IEA \
    --valid_tfrecord_fn /mnt/ceph/users/vgligorijevic/ContactMaps/data/Swiss-Model/merged_annot/tfrecords/*_chains_molecular_function_seqid_30_valid_EXP-IEA \
