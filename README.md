# Gender Bias in Coreference Resolution

## Introduction
This repo contains the code for the project 'Exploring Gender Bias in Coreference Resolution' for the course Deep Learning for Natural Language Processing at the University of Amsterdam. The code is organized in in jupyter notebooks and python scripts.

## Installation
To install the required packages, run the following command in the root directory of the project:
```
conda env create -n dl4nlp-pr --python=3.11
conda activate dl4nlp-pr
pip install -r requirements.txt
```

## Data
The data used in this project is the GAP dataset avaulable at the [official repo](https://github.com/google-research-datasets/gap-coreference). The data is already included in the repo under `data/gap/`, but can also be downloaded from the official repo. 
It contains the following files:
- `gap-development.tsv`
- `gap-test.tsv`
- `gap-validation.tsv`
We additionally include the modified version of the the gap test dataset where we replace the gendered pronouns with their gender-neutral counterparts. The modified dataset is available under `data/gap/gap-test-gn.tsv`. 

## Creating the ablated dataset
To create the ablated version of the GAP test dataset, which we will use for our experiments, run the cells in the notebook `modify-gap.ipynb`. The notebook will create the file `data/gap/gap-test-gn.tsv` which contains the ablated version of the GAP test dataset.

## Spanbert Evaluation in GAP

Here is the code, data and results for evaluating Spanbert in GAP data and NB GAP. \\

Note: Cuda hast to be lower than 12 to work with allennlp.


`spanbert_gap_eval.ipynb` is used to load the spanbert model or download it if its missing. We do some data processing, generate predictions for GAP data and use the official scorer from GAP repository to compute F1, accuracy and use our own custom function for scoring B3.

