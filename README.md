# Multi-tissue imputation

## Quick reference of main files
- `train_gtex.py`: Main script to train the multi-tissue imputation model on normalised GTEx data
- `evaluate_GTEx_v8_normalised.ipynb`: Analysis of multi-tissue imputation quality on normalised data (i.e. model trained via `train_gtex.py`)
- `evaluate_GTEx_v9_signatures_normalised.ipynb`: Analysis of cell-type signature imputation (i.e. fine-tunes model on GTEx-v9)

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

## Citation
If you use this code for your research, please cite our paper:
```
@article {Vinas2022Hypergraph,
	author = {Vinas Torne, Ramon and Joshi, Chaitanya K. and Dumitrascu, Bianca and Gamazon, Eric and Lio, Pietro},
	title = {Hypergraph factorisation for multi-tissue gene expression imputation},
	elocation-id = {2022.07.31.502211},
	year = {2022},
	doi = {10.1101/2022.07.31.502211},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/08/01/2022.07.31.502211},
	eprint = {https://www.biorxiv.org/content/early/2022/08/01/2022.07.31.502211.full.pdf},
	journal = {bioRxiv}
}
```
