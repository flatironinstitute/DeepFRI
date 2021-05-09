#!/usr/local/bin/python3

import argparse
import numpy as np
import seaborn as sns
import sys
import json

import matplotlib.pyplot as plt
plt.switch_backend('agg')


# draw class activation map
def draw_cam(window, cam, pdb_chain, goterm, sequence):
    #  Prepare the final display
    seq = [s for s in sequence]
    x = np.arange(len(seq))
    y = cam

    # draw the class activation map
    buf = '%s: Function = %s' % (pdb_chain, goterm)
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(x, y, color='gray', label='gradCAM')
    axes[0].set_title(buf, fontsize=16)
    axes[0].set_ylabel('Activation', fontsize=16)
    axes[0].set_xlim((0, len(seq)))
    axes[0].legend(fontsize=16)

    axes[1] = sns.heatmap(cam.reshape(1, cam.shape[0]), cmap='bwr', alpha=0.8, cbar=False)
    axes[1].set_xlabel("Residues", fontsize=16)
    axes[1].set_xlim((0, len(seq)))
    axes[1].set_yticks([])
    axes[1].set_xticks([])
    plt.axis('tight')
    plt.savefig("saliency_fig_" + pdb_chain + "_" + goterm.replace(':', '.') + ".png")


# average over window, window has to be odd
def window_avg(values, window_size):

    # check for odd window
    if window_size % 2 == 0:
        sys.error("window length has to be odd number")

    half = int((window_size-1)/2)

    # values are a list of floats
    lengthened = values.copy()

    # add half a window to beginning with value of the first element
    for i in range(0, half):
        lengthened.insert(0, values[0])

    # add half a window to beginning with value of the last element
    for i in range(0, half):
        lengthened.append(values[-1])

    avg = []
    # sum values over list, creating a new list
    for i in range(0, len(values)):

        this_sum = 0
        for j in range(0, window_size):
            this_sum += lengthened[i+j]
        avg.append(this_sum / window_size)

    return avg

#  =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--saliency_fn', '-i', type=str, required=True, help='JSON file with saliency maps.')
    parser.add_argument('--list_all', '-l', action='store_true', help="List all protein IDs and their predicted functions.")
    parser.add_argument('--protein_id', '-p', type=str, default=None, help='Protein ID to be visualized.')
    parser.add_argument('--go_id', '-go', type=str, default=None, help='Example GO term, protein chains of which are to be visualized.')
    parser.add_argument('--go_name', type=str, default=None, help='Example GO term keyword, protein IDs of which are to be visualized.')
    parser.add_argument('--window', '-w', type=int, default=3, help='Sliding window length. Default 3.')

    args = parser.parse_args()
    pred = json.load(open(args.saliency_fn, 'r'))

    # if listing all
    if args.list_all:
        for chain in pred:
            for i in range(0, len(pred[chain]['GO_ids'])):
                print (chain, pred[chain]['GO_ids'][i], pred[chain]['GO_names'][i])
    else:

        chain_id = args.protein_id
        go_id = args.go_id
        go_name = args.go_name
        window = args.window

        lchains = []
        lgoterm = []
        lgoname = []
        lseq = []
        lcam = []

        # sort all into lists
        for chain in pred:
            for gt in range(0, len(pred[chain]['GO_ids'])):

                goterm = pred[chain]['GO_ids'][gt]
                goname = str(pred[chain]['GO_names'][gt])
                seq = pred[chain]['sequence']
                cam = pred[chain]['saliency_maps'][gt]

                if (chain_id is not None and chain == chain_id) and (go_id is not None and go_id == goterm):
                    lchains.append(chain)
                    lgoterm.append(goterm)
                    lgoname.append(goname)
                    lseq.append(seq)
                    lcam.append(cam)
                if (chain_id is not None and chain == chain_id) and (go_name is not None and goname.find(go_name) != -1):
                    lchains.append(chain)
                    lgoterm.append(goterm)
                    lgoname.append(goname)
                    lseq.append(seq)
                    lcam.append(cam)
                if (chain_id is not None and chain == chain_id) and (go_name is None) and (go_id is None):
                    lchains.append(chain)
                    lgoterm.append(goterm)
                    lgoname.append(goname)
                    lseq.append(seq)
                    lcam.append(cam)
                if (go_id is not None and go_id == goterm) and (chain_id is None) and (go_name is None):
                    lchains.append(chain)
                    lgoterm.append(goterm)
                    lgoname.append(goname)
                    lseq.append(seq)
                    lcam.append(cam)
                if (go_name is not None and goname.find(go_name) != -1) and (chain_id is None) and (go_id is None):
                    lchains.append(chain)
                    lgoterm.append(goterm)
                    lgoname.append(goname)
                    lseq.append(seq)
                    lcam.append(cam)

        if args.protein_id is not None and chain_id not in lchains:
            raise ValueError("Protein ID not in the list.")

        if args.go_id is not None and go_id not in lgoterm:
            raise ValueError("GO ID not in the list.")

        # go through the picked ones
        c = 0

        # os.system( "rm pymol_viz.py" )
        with open('pymol_viz.py', 'w') as f:
            f.write("#!/usr/bin/env python\n")
            f.write("import time\n")
            f.write("import pymol\n\n")
            f.write("pymol.cmd.bg_color( \'white\' )\n")
            f.write("pymol.cmd.viewport( \'2000\', \'2000\' )\n")
            f.write("pymol.cmd.set( \'sphere_scale\', \'0.5\' )\n")

            for c in range(0, len(lchains)):
                cam_values = lcam[c]
                new_cam = window_avg(cam_values, window)
                draw_cam(window, np.array(new_cam), lchains[c], lgoterm[c], lseq[c])

                # write script to run in pymol
                f.write("pymol.cmd.fetch( \'" + lchains[c] + "\' )\n")
                f.write("time.sleep( 1 )\n")
                f.write("pymol.cmd.show( \'cartoon\' )\n")

                f.write("pymol.cmd.set( \'sphere_scale\', \'1\' )\n")

                # alter b-factor columns with custom coloring
                cam_string = str(", ".join(map(str, new_cam)))
                f.write("pymol.cmd.alter(\'" + lchains[c] + "\', \'b=0.0\' )\n")
                f.write("stored.cam = [" + cam_string + "]\n")
                f.write("pymol.cmd.alter( \'" + lchains[c] + " and n. CA\', \'b=stored.cam.pop(0)\' )\n")

                # a bunch of nice color schemes
                # f.write( "pymol.cmd.spectrum(\'b\', \'slate_yellow_red\', \'" + lchains[c] + " and n. CA\' )\n")
                # f.write( "pymol.cmd.spectrum(\'b\', \'silver_yellow_red\', \'" + lchains[c] + " and n. CA\' )\n")
                # f.write( "pymol.cmd.spectrum(\'b\', \'palecyan_silver_magenta\', \'" + lchains[c] + " and n. CA\' )\n")
                # f.write( "pymol.cmd.spectrum(\'b\', \'aquamarine_silver_red\', \'" + lchains[c] + " and n. CA\' )\n")
                # f.write( "pymol.cmd.spectrum(\'b\', \'lightblue_silver_red\', \'" + lchains[c] + " and n. CA\' )\n")
                f.write("pymol.cmd.spectrum( \'b\', \'blue_white_red\', \'" + lchains[c] + " and n. CA\' )\n")

                # ligands, water, etc
                f.write("pymol.cmd.select( \'ligs_" + lchains[c] + "\', \'" + lchains[c] + " and het\' )\n")
                f.write("pymol.cmd.select( \'water\', \'resn hoh\' )\n")
                f.write("pymol.cmd.remove( \'water\' )\n")
                f.write("pymol.cmd.show( \'sphere\', \'ligs_" + lchains[c] + "\' )\n")
                f.write("pymol.cmd.color( \'yellow\', \'ligs_" + lchains[c] + "\' )\n")
                f.write("\n")
        f.close()
