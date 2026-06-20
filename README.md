# HotRelax: High-order tensor neural network for iteration-free structure relaxation

This repository contains the official PyTorch implementation of the work "HotRelax: High-order tensor neural network for iteration-free structure relaxation".
We provide the code for training the base model setting on the X-Mn-O, MP, C2DB, 2DMD and OC20 datasets.

## Content ##
0. [Environment Setup](#environment-setup)
0. [Data Preprocessing](#data-preprocessing)
0. [Training](#training)
0. [Evaluation](#evaluation)
0. [File Structure](#file-structure)
0. [Citation](#citation)
0. [Acknowledgement](#acknowledgement)



## Environment Setup ##

- Use conda to create a new environment named `hotrelax`:
```
conda env create -n hotrelax python==3.10
```

- Please ensure that your Python version is at least 3.8.

- Activate the environment:
```
conda activate hotrelax
```

- Then install the required package versions:
```
pip install -r requirements.txt
```


## Data Preprocessing

### Download

The datasets can be downloaded from [Zenodo](https://zenodo.org/records/20772345). After unzipping them, you can find the preprocessed data in the following directories for each dataset:

- For the XMnO dataset: `data_xmno/train`, `data_xmno/val`, `data_xmno/test`
- For the MP and 2DMD datasets: the directory structure is similar to that of XMnO
- For the OC20 dataset:
    - two subsets for training: `data_oc20/train`, `data_oc20/val`
    - four subsets for validation: `data_oc20/val_id`, `data_oc20/val_ood_ads`, `data_oc20/val_ood_cat`, `data_oc20/val_ood_both`
- For the C2DB dataset: this dataset is available upon request. Please contact the corresponding author of the C2DB dataset to obtain the files.


### Transforming Data into `.pt` Format [Optional]

We recommend converting the data into `.pt` format to speed up preprocessing during training. Use the following command, and replace `your_data_path` with the actual path to your data:
```
python transform.py your_data_path
```


## Training ##

Train HotRelax by running:

```
python train.py --input_file input.yaml
```
For different datasets, you only need to modify `input.yaml`. The hyperparameters that may need to be adjusted include `trainBatch`, `testBatch`, `trainSet`, `testSet`, `evalSet`, `numWorkers`, and `elements`.
For convenience, we provide input files for each dataset in the [`configs`](configs) folder.


## Evaluation ##

After training, the best model will be saved at `./outDir/best.pt`. You can evaluate HotRelax by running:
```
python eval.py --input_file input.yaml --model_file ./outDir/best.pt
```
This command will generate a result file named `results.txt`, which includes the coordinate MAE, lattice MAE, match rate, and inference time for each structure.


## File Structure ##

1. [`hotrelax`](hotrelax) contains the implementation of the HotRelax architecture.
2. [`train.py`](train.py) contains the training code.
3. [`eval.py`](eval.py) contains the evaluation code.
4. [`configs`](configs) contains configuration files for training and evaluation.


## Citation ##

Please consider citing the works below if this repository is helpful:

- [HotPP](https://doi.org/10.1038/s41467-024-51886-6):
    <br>
@article{Wang2024,  
  &emsp;&emsp;author = {Junjie Wang and Yong Wang and Haoting Zhang and Ziyang Yang and Zhixin Liang and Jiuyang Shi and  	Hui-Tian Wang and Dingyu Xing and Jian Sun},  
  &emsp;&emsp;title = {E(n)-Equivariant cartesian tensor message passing interatomic potential},  
  &emsp;&emsp;journal = {Nature Communications},  
  &emsp;&emsp;volume = {15},
  &emsp;&emsp;number = {1},  
  &emsp;&emsp;pages = {7607},  
  &emsp;&emsp;year = {2024},  
  &emsp;&emsp;month = {September},  
  &emsp;&emsp;doi = {10.1038/s41467-024-51886-6},  
  &emsp;&emsp;url = {https://doi.org/10.1038/s41467-024-51886-6},  
  &emsp;&emsp;issn = {2041-1723}  
}<br>  


## Acknowledgement ##

Our implementation is based on [PyTorch](https://pytorch.org/), [HotPP](https://gitlab.com/bigd4/hotpp).