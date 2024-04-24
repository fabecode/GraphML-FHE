# XGBoost with Graph Feature Preprocessor FHE Pipeline
This directory contains the code for my Final Year Project, where privacy-preserving Fully Homomorphic Encryption over the Torus (TFHE) is applied to XGBoost using Concrete-ML. The anti-money laundering features were preprocessed using IBM's Snap ML [Graph Feature Preprocessor](https://snapml.readthedocs.io/en/latest/graph_preprocessor.html) to produce graph-based features.

## Setup
To use the repository, you first need to install the conda environment via 
```
conda env create -f env.yml python=3.9
```
Note that concrete-ml requires torch 1.13.1 to work.

The data needed for the experiments can be found on [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data). To use this data with the provided training scripts, you first need to perform a pre-processing step:
```
python format_kaggle_files.py '/path/to/kaggle-file/' '/path/to/output-file/'
```

## Usage
To run the experiments, run `main.py` 

## Licence
Apache License
Version 2.0, January 2004