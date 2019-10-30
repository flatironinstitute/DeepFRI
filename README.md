# DeepFRIer
Deep functional residue identification
<img src="figs/pipeline.png">

## Citing
```
@article {Gligorijevic2019,
	author = {Gligorijevic, Vladimir and Renfrew, P. Douglas and Kosciolek, Tomasz and Leman,
	Julia Koehler and Cho, Kyunghyun and Vatanen, Tommi and Berenberg, Daniel
	and Taylor, Bryn and Fisk, Ian M. and Xavier, Ramnik J. and Knight, Rob and Bonneau, Richard},
	title = {Structure-Based Function Prediction using Graph Convolutional Networks},
	year = {2019},
	doi = {10.1101/786236},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2019/10/04/786236},
	journal = {bioRxiv}
}

```
## Dependencies

*DeepFRIer* is tested to work under Python 3.6.

The required dependencies for *deepFRIer* are [Keras](https://keras.io/), [TensorFlow](https://www.tensorflow.org/), [Numpy](http://www.numpy.org/) and [scikit-learn](http://scikit-learn.org/).


# Protein function prediction
To predict functions of a protein use **predict.py** script with the following FLAGS:

* `seq`             str, Protein sequence as a string
* `cmap`            str, Name of a file storing protein contact map and sequence in `*.npz` file format
                    (with the following numpy array variables: `A_ca`, `sequence`, `L`)
* `cmap_csv`        str, Filename of the catalogue (in `*.csv` file format) containg mapping between protein names and directory of `*.npz` filenames
* `fasta_fn`        str, Fasta filename
* `output_fn_prefix`str, Output filename for saving predictions and class
                    activation maps.
* `verbose`         bool, Whether or not to print function prediction results.
* `saliency`        bool, Whether or not to compute class activaton maps.


## Example:

Predicting MF-GO terms for Parvalbumin alpha protein (PDB: [1S3P](https://www.rcsb.org/structure/1S3P)):

```
>> python predict.py --cmap 1S3P-A.npz --verbose

```

## Output:


```txt
Protein GO-term/EC-number Score GO-term/EC-number name
query_prot GO:0043167 0.95134 ion binding
query_prot GO:0046872 0.90832 metal ion binding
query_prot GO:0043169 0.90517 cation binding
query_prot GO:0005509 0.87179 calcium ion binding
query_prot GO:0043168 0.06332 anion binding
query_prot GO:0031072 0.00247 heat shock protein binding
query_prot GO:1901567 0.00099 fatty acid derivative binding
query_prot GO:0045159 0.00009 myosin II binding
query_prot GO:0032027 0.00001 myosin light chain binding

```



# Training DeepFRI
To train *deepFRI* run the following command from the project directory:
```
>> python train_DeepFRI.py --model_name model_name_prefix
```

## Output
Generated files:
* `model_name_prefix_model.hdf5`   trained model with architecture and weights saved in HDF5 format
* `model_name_prefix_pred_scores.pckl` pickle file with predicted GO term/EC number scores for test proteins

# Flags

A number of FLAGS is available to specify the behavior of *deepFRI*, both for prediction and training:

* `model_name`      str, name of the model. Default: `GCN-LM_model`
* `gcn_dims`        list (int), dimensions of GCN layers. Default: `[128, 256, 512]`
* `hidden_dims`	    list (int), dimensions of Dense layers. Default: `[512]`
* `dropout`	    float, dropout rate for Dense layer. Default: `0.30`
* `l2_reg` 	    float, l2 regularization coefficient for GCN layers. Default: `1e-4`
* `epochs`          int, number of epochs to train the model. Default: `100`
* `batch_size`	    int, Batch size. Default: `64`
* `pad_len`         int, maximum padding length for sequences and contact maps. Default: `1000`
* `results_dir`     str, directory with exported models and results. Default: `./results/`
* `ont`             str, GO or EC ontology. Default: `molecular function`
* `cmap_type`       str, type of contact maps (A_nbr, A_ca or A_all). Default: `A_ca`
* `lm_model_name`   str, keras pre-trained LSTM Language Model name. Default: `./trained_models/lstm_lm.h5`
* `split_fn`        str, pickle file with train/test/valid PDB IDs and their annotatin matrix.
		    Default: `train_test_split_seqsim_30.pckl`
* `catalogue`       str, csv file mapping PDB IDs to numpy files storing individual contact maps. Default: `catalogue.csv`
* `train_tfrecord_fn`	str, train tfrecords file name. Default: `train.tfrecords`
* `valid_tfrecord_fn`	str, validaiton tfrecords file name. Default: `valid.tfrecords`

## Data

Data (*train_tfrecord*, *valid_tfrecord* files) used for producing figures in the paper can be downloaded from:

https://users.flatironinstitute.org/vgligorijevic/public_www/deepFRIer

# Functional residue identification
<img src="figs/saliency.png">





