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
    parser.add_argument('--use_guided_grads', help="Prints predictions.", action="store_true")
    parser.add_argument('--saliency', help="Compute saliency maps for every protein and every MF-GO term.", action="store_true")
    args = parser.parse_args()

    if args.seq is not None or args.fasta_fn is not None:
        gcn = False
        layer_name = "CNN_concatenate"
        models = {"ec": "./results/DeepCNN-MERGED_enzyme_commission",
                  "mf": "./results/DeepCNN-MERGED_molecular_function",
                  "bp": "./results/DeepCNN-MERGED_biological_process",
                  "cc": "./results/DeepCNN-MERGED_cellular_component"
                  }

    elif args.cmap is not None or args.cmap_csv is not None:
        gcn = True
        layer_name = "GCNN_concatenate"
        models = {"mf": "./results/DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_molecular_function",
                  "bp": "./results/DeepFRI-MERGED_MultiGraphConv_3x512_fcd_2048_ca_10A_biological_process",
                  "cc": "./results/DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_cellular_component",
                  'ec': "./results/DeepFRI-MERGED_MultiGraphConv_3x512_fcd_1024_ca_10A_enzyme_commission"
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
        # predictor.save_predictions(args.output_fn_prefix + "_" + ont.upper() + "_pred_scores.pckl")

        if args.saliency and ont in ['mf', 'ec']:
            predictor.compute_GradCAM(layer_name=layer_name, use_guided_grads=args.use_guided_grads)
            predictor.save_GradCAM(args.output_fn_prefix + "_" + ont.upper() + "_saliency_maps.pckl")
