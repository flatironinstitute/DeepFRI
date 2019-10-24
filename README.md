# DeepFRIer
Deep functional residue identification
<img src="figs/pipeline.png">

# Citing
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

## Data

Data (*tfrecord* train/validation files) used for producing figures in the paper can be downloaded from:

https://users.flatironinstitute.org/vgligorijevic/public_www/deepFRIer

## Training DeepFRI
To train *deepFRI* run the following command from the project directory:
```
python train_DeepFRI.py --model_name model_name
```

# FLAGS

A number of FLAGS is available to specify the behavior of *deepFRI*, both for prediction and training:

* `gcn_dims`        list (int), dimensions of GCN layers
* `hidden_dims`	    list (int), dimensions of Dense layers
* `dropout`	    float, dropout rate for Dense layer
* `l2_reg` 	    float, l2 regularization coefficient for GCN layers
* `epochs`          int, number of epochs to train the model. Default: `100`
