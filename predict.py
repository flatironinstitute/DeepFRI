import argparse
from deepfrier.Predictor import Predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seq', type=str,  help="Protein sequence to be annotated.")
    parser.add_argument('-cm', '--cmap', type=str,  help="Protein contact map to be annotated (in *npz file format).")
    parser.add_argument('--cmap_csv', type=str,  help="Catalogue with chain to file path mapping.")
    parser.add_argument('--fasta_fn', type=str,  help="Fasta file with protein sequences.")
    parser.add_argument('--model_fn_prefix', type=str, default='GCN_molecular_function', help="Name of the GCN/CNN model.")
    parser.add_argument('-ont', '--ontology', type=str, default=['mf'], nargs='+', choices=['mf', 'bp', 'cc', 'ec'], help="Gene Ontology/Enzyme Commission.")
    parser.add_argument('-o', '--output_fn_prefix', type=str, default='DeepFRI', help="Save predictions/saliency in file.")
    parser.add_argument('-v', '--verbose', help="Prints predictions.", action="store_true")
    parser.add_argument('--saliency', help="Compute saliency maps for every protein and every MF-GO term.", action="store_true")
    args = parser.parse_args()

    if args.seq is not None or args.fasta_fn is not None:
        gcn = False
        layer_name = "CNN_layer"
        models = {"ec": "./results/CNN-1HOT_MERGED-enzyme_commission_EXP-IEA_seqid_95_filter_nums_8x512_filter_lens_8-64_softmax_mixed_test",
                  "mf": "./results/CNN-1HOT_MERGED-molecular_function_EXP-IEA_seqid_95_filter_nums_16x512_filter_lens_8-128_softmax_mixed_test",
                  "bp": "./results/CNN-1HOT_MERGED-biological_process_EXP-IEA_seqid_95_filter_nums_16x512_filter_lens_8-128_softmax_mixed_test",
                  "cc": "./results/CNN-1HOT_MERGED-cellular_component_EXP-IEA_seqid_95_filter_nums_16x512_filter_lens_8-128_softmax_mixed_test"
                  }

    elif args.cmap is not None or args.cmap_csv is not None:
        gcn = True
        layer_name = "GCNN_concatenate"
        models = {"mf": "./results/GCN-LM_MERGED_molecular_function_EXP-IEA_seqid_95_gcn_256-256-512_hidd_1024_softmax_mixed_test",
                  "bp": "./results/GCN-LM_MERGED_biological_process_EXP-IEA_seqid_95_gcn_128-256-256-512_hidd_1024_softmax_mixed_test",
                  "cc": "./results/GCN-LM_MERGED_cellular_component_EXP-IEA_seqid_95_gcn_128-128-256_hidd_512_softmax_mixed_test",
                  'ec': "./results/GCN-LM_MERGED_enzyme_commission_EXP-IEA_seqid_95_gcn_256-256-512_hidd_800_softmax_mixed_test"
                  }

    for ont in args.ontology:
        predictor = Predictor(models[ont], gcn=gcn)
        if args.seq is not None:
            predictor.predict(args.seq)
        if args.cmap is not None:
            prot_name = args.cmap
            prot_name = prot_name.split('/')[-1].split('.')[0]
            predictor.predict(args.cmap, chain=prot_name)
        if args.fasta_fn is not None:
            predictor.predict_from_fasta(args.fasta_fn)
        if args.cmap_csv is not None:
            predictor.predict_from_catalogue(args.cmap_csv)
        predictor.export_csv(args.output_fn_prefix + "_" + ont.upper() + "_predictions.csv", args.verbose)

        if args.saliency and ont in ['mf']:
            predictor.compute_gradCAM(layer_name=layer_name)
            predictor.save_gradCAM(args.output_fn_prefix + "_" + ont.upper() + "_saliency_maps.pckl")
