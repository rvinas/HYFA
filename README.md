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

## Installation
1. Clone this repository.
2. Install the dependencies via the following command:
```pip install -r requirements.txt```. The installation typically takes a few minutes.

## Running the model
1. Prepare your dataset:
   * By default, the script `train_gtex.py` loads a dataset from a CSV file (`GTEX_FILE`) with the following format:
     * Columns are genes and rows are samples.
     * Entries correspond to normalised gene expression values.
     * The first row has the gene identifiers.
     * The first column has the donor identifiers. The file might contain multiple rows per donor.
     * An extra column `tissue` denotes the tissue from which the sample was collected. The combination of donor and tissue identifier is unique.
   * The metadata is loaded from a separate CSV file (`METADATA_FILE`; see function `GTEx_metadata` in `train_gtex.py`). Rows correspond to donors and columns to covariates. By default, the script expects at least two columns: `AGE` (integer) and `SEX` (integer).

2. Run the script `train_gtex.py` to train HYFA. This uses the default hyperparameters from `config/default.yaml`. After training, the model will be stored in your current working directory.

3. Once the model is trained, evaluate your results via the notebook `evaluate_GTEx_v8_normalised.ipynb`.


<!--- The function `GTEx_v8_normalised_adata` populates an [`AnnData`](https://anndata.readthedocs.io/en/latest/) object. --->

## Citation
If you use this code for your research, please cite our paper:
```
@article {Vinas2022Hypergraph,
	author = {Vinas Torne, Ramon and Joshi, Chaitanya K. and Georgiev, Dobrik and Dumitrascu, Bianca and Gamazon, Eric and Lio, Pietro},
	title = {Hypergraph factorisation for multi-tissue gene expression imputation},
	elocation-id = {2022.07.31.502211},
	year = {2022},
	doi = {10.1101/2022.07.31.502211},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/10.1101/2022.07.31.502211v3},
	eprint = {https://www.biorxiv.org/content/10.1101/2022.07.31.502211v3.full.pdf},
	journal = {bioRxiv}
}
```
