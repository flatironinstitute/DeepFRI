import argparse
from deepfrier.Predictor import Predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seq', type=str,  help="Protein sequence to be annotated.")
    parser.add_argument('-cm', '--cmap', type=str,  help="Protein contact map to be annotated (in *npz file format).")
    parser.add_argument('--cmap_csv', type=str,  help="Catalogue with chain to file path mapping.")
    parser.add_argument('--fasta_fn', type=str,  help="Fasta file with protein sequences.")
    parser.add_argument('--model_fn_prefix', type=str, default='GCN_molecular_function', help="Name of the GCN/CNN model.")
    parser.add_argument('-o', '--output_fn_prefix', type=str, default='DeepFRI', help="Save predictions/saliency in file.")
    parser.add_argument('-v', '--verbose', help="Prints predictions.", action="store_true")
    parser.add_argument('--saliency', help="Compute saliency maps for every protein.", action="store_true")
    args = parser.parse_args()

    if args.seq is not None or args.fasta_fn is not None:
        gcn = False
        layer_name = "CNN_layer"
        models = {"Enzyme Commission": "./trained_models/CNN-1HOT_SWISS-enzyme_commission_EXP_seqid_90_filter_nums_120-100-80-60_filter_lens_5-10-15-20",
                  "Molecular Function": "./trained_models/CNN-1HOT_SWISS-molecular_function_EXP-IEA_seqid_90_filter_nums_300-200-150-100_filter_lens_5-10-15-20"}
        # models = {"Enzyme Commission": "./sifts_models/CNN-1HOT_SIFTS-enzyme_commission_EXP_seqid_30_filter_nums_120-100-80-60_filter_lens_5-10-15-20",
        #          "Molecular Function": "./sifts_models/CNN-1HOT_SIFTS-molecular_function_EXP-IEA_seqid_30_filter_nums_300-200-150-100_filter_lens_5-10-15-20"}

    elif args.cmap is not None or args.cmap_csv is not None:
        gcn = True
        layer_name = "GCNN_concatenate"
        models = {"Enzyme Commission": "./trained_models/GCN-LM_SWISS-enzyme_commission_EXP_seqid_90_gcn_128-256-512_hidd_1024",
                  "Molecular Function": "./trained_models/GCN-LM_SWISS-molecular_function_EXP-IEA_seqid_90_gcn_128-256-512_hidd_850"}
        # models = {"Enzyme Commission": "./sifts_models/GCN-LM_SIFTS-enzyme_commission_EXP_seqid_30_gcn_128-128-256_hidd_512",
        #          "Molecular Function": "./sifts_models/GCN-LM_SIFTS-molecular_function_EXP-IEA_seqid_30_gcn_128-256-512_hidd_750"}

    ec_predictor = Predictor(models['Enzyme Commission'], gcn=gcn)
    mf_predictor = Predictor(models['Molecular Function'], gcn=gcn)
    if args.seq is not None:
        ec_predictor.predict(args.seq)
        mf_predictor.predict(args.seq)
    if args.cmap is not None:
        ec_predictor.predict(args.cmap)
        mf_predictor.predict(args.cmap)
    if args.fasta_fn is not None:
        ec_predictor.predict_from_fasta(args.fasta_fn)
        mf_predictor.predict_from_fasta(args.fasta_fn)
    if args.cmap_csv is not None:
        ec_predictor.predict_from_catalogue(args.cmap_csv)
        mf_predictor.predict_from_catalogue(args.cmap_csv)

    ec_predictor.export_csv(args.output_fn_prefix + "_EC_predictions.csv", args.verbose)
    mf_predictor.export_csv(args.output_fn_prefix + "_MF_predictions.csv", args.verbose)

    if args.saliency:
        mf_predictor.compute_gradCAM(layer_name=layer_name)
        mf_predictor.save_gradCAM(args.output_fn_prefix + "_MF_saliency_maps.pckl")
