# Self-Supervised-Sketch2Image-pytorch
A pytorch implementation of self-supervised sketch-to-image model, the paper can be found [here](https://arxiv.org/abs/2012.09290).

## 0. Data
For CelebA and WikiArt paintings image, the pre-processed RGB image data and their corresponding sketch images are available at this [link](https://drive.google.com/drive/folders/1nBvan8xnEMQpM39sJYd1fH0B-QFdzRxK?usp=sharing)

## 1. Description
The code is structured as follows:
* models.py: all the models' structure definition, including style encoder, content encoder, decoder, generator, and discriminator.

* datasets.py: the data pre-processing and data loading methods during training, such as the style image augmenting logic and the sketch augmenting logic.

* train_step_1_ae.py: the training details of the autoencoder, including the objective functions, optimization methods, training procedures.

* train_step_2_gan.py: the training details of the GAN, including the objective functions, optimization methods, training procedures.

* train.py: the main entry of the code, execute this file to train our model, the intermediate results and checkpoints will be saved periodically.

* config.py: the settings of all the hyper-parameters are defined here, such as learning rate, batch-size, and image folder paths.

* benchmark.py: the functions we used to compute FID are located here, it automatically downloads the pytorch official inception model. 

* lpips: this folder contains the code to compute the LPIPS score, the inception model is also automatically download from official location.

* sketch_styletransfer: this folder contains the code to train the proposed sketch synthesis model, all the detailed configs can be found in sketch_styletransfer/train.py.

## 2. How to run
### 2.1 Synthesis sketches with TOM 
To train TOM for synthesizing sketches for an RGB-image dataset, put all your RGB-images in a folder, and place all you rcollected sketches into another folder
```
cd sketch_styletransfer
python train.py --path_a /path/to/RGB-image-folder --path_b /path/to/real-sketches
```
You can also see all the training options by:
```
python train.py --help
```
The code will automatically create a new folder to store all the trained checkpoints and intermediate synthesis results.

Once finish training, you can generate sketches for all RGB images in your dataset use the saved checkpoints:
```
python evaluate.py --path_content /path/to/RGB-image-folder --path_result /your/customized/path/for/synthesized-sketches --checkpoint /choose/any/checkpoint/you/like
```

### 2.2 Train the Sketch-to-image model

Training the main model is as simple as
```
python train.py 
```
This code will automatically conduct the training of both the AutoEncoder and GAN (first train an AE then for a GAN, as described in the paper). A folder is automatically created and save intermediate checkpoints and generated images.

The benchmarking FID on training set is also printed in the terminal every fix amount of iterations.


#### 2.2.1 Dataset
This code is ready to train on your own image datasets. And training on datasets used in the model (CelebA and WikiArt) or on your own datasets are the same: just place all images from a dataset into one folder.

#### 2.2.2 Config training
To config the training, just edit the config.py. Note that there are three available "DATA_NAME" to choose from: "art", "face" and "shoe". "art" is the specified optimal structure for WikiArt dataset, "face" is for CelebA, and "shoe" is for small datasets with only few thousand RGB-images (or even less). When train on your own dataset, you can try with the three options and see which one works the best. Note that the "shoe" option will employ an extra training objective for the AutoEncoder to enforce the effectiveness of the Content-Encoder.

### 2.3 Evaluation
Running benchmarks is as simple as
```
python benchmark.py
```
Which you should specify the model path and image folder path inside the code.

We also provide code for generating images as we displayed in the paper, including generating style-transfer, style-mixing and sketch-to-image results.
All the code are located in folder "evaluate". 




## 3. Extra notes
The provided code is for research use only, and is a simplified version from what described in the paper. Only few changes are made due to business concerns (we developed this model for commercial use).
* The code only uses 3 synthesised paired-sketches for each RGB-image, instead of 10 as described in the paper.
* The code ommits the mutual information minimization objective for the content-encoder.
* The code defines the models with some more refined structures such as Swish activation instead of ReLU, and Resblock that is not mentioned in the paper.

Despite the changes, the code still able to train a model that beats state-of-the-art models just as we claimed in the paper, and only slightly worse than our fully-flaged version. Importantly, we believe one can easily re-implement the ommitted parts mentioned above.
