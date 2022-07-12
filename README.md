# Multi-tissue imputation

## Quick reference of main files
- `train_gtex.py`: Main script to train the multi-tissue imputation model on normalised GTEx data
- `train_GTEx_v8_normalised.ipynb`: Analysis of multi-tissue imputation quality on normalised data (i.e. model trained via `train_gtex.py`)
- `train_GTEx_v9_deconvolution.ipynb`: Analysis of cell-type signature imputation (i.e. fine-tunes model on GTEx-v9)

## Data
- `src/data.py`: Data object encapsulating multi-tissue gene expression
- `src/dataset.py`: Dataset that takes care of processing the data
- `src/data_utils.py`: Data utilities

## Model
- `src/hnn.py`: Hypergraph neural network
- `src/hypergraph_layer.py`: Message passing on hypergraph
- `src/hnn_utils.py`: Hypergraph model utilities
- `src/metagene_encoders.py`: Model transforming gene expression to metagene values
- `src/metagene_decoders.py`: Model transforming metagene values to gene expression

## Training
- `src/train_utils.py`: Train/eval loops
- `src/distribions.py`: Count data distributions
- `src/losses.py`: Loss functions for different data likelihoods

## Other utils
- `src/pathway_utils.py`: Utilities to retrieve KEGG pathways
- `src/ct_signature_utils.py`: Utilities for inferring cell-type signatures
