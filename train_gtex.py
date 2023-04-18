"""
Trains the model on GTEx data
"""
import torch
import numpy as np
import pandas as pd
import wandb
import argparse
from torch.utils.data import Dataset, DataLoader

from src.hnn import HypergraphNeuralNet
from src.data import Data
from src.dataset import HypergraphDataset
from src.data_utils import *
from src.eval_utils import *
from src.train_utils import train
import scanpy as sc

np.random.seed(0)
num_workers = 4

GTEX_FILE = 'data/GTEX_data.csv'
METADATA_FILE = 'data/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt'

def GTEx(file=GTEX_FILE):
    """
    Loads processed GTEx data
    :param file: path of the CSV file
    :return: Returns
        - data: numpy array of shape=(nb_samples, nb_genes)
        - gene_symbols: numpy array with gene symbols. Shape=(nb_genes,)
        - sampl_ids: numpy array with sample IDs (GTEx IDs of individuals, e.g. GTEX-1117F). Shape=(nb_samples,)
        - tissues: numpy array indicating the tissue of each sample. Shape=(nb_samples,)
    """
    # Load data
    df = pd.read_csv(file, index_col=0)  # .sample(frac=1, random_state=random_seed)
    tissues = df['tissue'].values
    sampl_ids = df.index.values
    del df['tissue']
    data = np.float32(df.values)
    gene_symbols = df.columns.values
    return data, gene_symbols, sampl_ids, tissues


def GTEx_metadata(file=METADATA_FILE):
    """
    Loads metadata DataFrame with information about individuals
    :param file: path of the file
    :return: Pandas DataFrame with subjects' information
    """
    df = pd.read_csv(file, delimiter='\t')
    df = df.set_index('SUBJID')
    return df


def GTEx_v8_normalised_adata(file=GTEX_FILE):
    data, gene_symbols, sampl_ids, tissues = GTEx(file=file)
    metadata_df = GTEx_metadata()

    adata = sc.AnnData(data)
    adata.var['Symbol'] = gene_symbols
    adata.obs['Participant ID'] = sampl_ids
    adata.obs['Tissue'] = tissues

    # Delete participants with only one measured tissue
    adata = adata[adata.obs['Participant ID'].duplicated(keep=False)]

    # Static keys
    adata.obs['Tissue_idx'], tissue_dict = map_to_ids(adata.obs['Tissue'].values)
    adata.uns['Tissue_dict'] = tissue_dict
    # del adata.obs['Tissue']

    # Dynamic keys
    adata.obs['Participant ID_dyn'] = adata.obs['Participant ID']

    # Populate participant features
    adata.obs['Age'] = [float(a[:2]) for a in metadata_df.loc[adata.obs['Participant ID']]['AGE'].values]
    adata.obs['Sex'] = metadata_df.loc[adata.obs['Participant ID']]['SEX'].values-1
    donor_age = adata.obs['Age'] / 100
    donor_sex, donor_sex_dict = map_to_ids(adata.obs['Sex'])
    adata.obsm['Participant ID_feat'] = np.stack((donor_age, donor_sex), axis=-1)
    adata.uns['Sex_dict'] = donor_sex_dict

    # Put gene expression in layer
    adata.layers['x'] = adata.X

    return adata

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='configs/default.yaml', type=str)
    args, unknown = parser.parse_known_args()

    # Initialise wandb
    wandb.init(project='multitissue_imputation', entity="multitissue_imputation_project", config=args.config)
    config = wandb.config
    print(config)

    # Load data
    adata = GTEx_v8_normalised_adata()

    # Dictionaries
    _, tissue_dict = map_to_ids(adata.obs['Tissue'])
    tissue_dict_inv = {v: k for k, v in tissue_dict.items()}

    # Split train/val/test
    donors = adata.obs['Participant ID'].values
    train_donors = np.loadtxt('data/splits/gtex_train.txt', delimiter=',', dtype=str)
    val_donors = np.loadtxt('data/splits/gtex_val.txt', delimiter=',', dtype=str)
    test_donors = np.loadtxt('data/splits/gtex_test.txt', delimiter=',', dtype=str)
    train_mask = np.isin(donors, train_donors)
    test_mask = np.isin(donors, test_donors)
    val_mask = np.isin(donors, val_donors)
    print(len(train_donors), len(val_donors), len(test_donors))

    collate_fn = Data.from_datalist
    dtype = torch.float32  # torch.double
    target_tissues = ['Lung', 'Pancreas', 'Heart_Atrial', 'Esophagus_Muscularis']
    source_tissues = [t for t in adata.obs['Tissue'].unique() if t not in target_tissues]  # All tissues except targets

    train_dataset = HypergraphDataset(adata[train_mask], dtype=dtype, disjoint=True, static=False)
    val_dataset = HypergraphDataset(adata[val_mask], dtype=dtype, disjoint=False, static=True, obs_source={'Tissue': source_tissues}, obs_target={'Tissue': target_tissues})
    # test_dataset = HypergraphDataset(adata[test_mask], dtype=dtype, static=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=num_workers)

    # device = torch.device("cpu")
    # Use certain GPU
    device = torch.device("cuda:{}".format(config.gpu) if torch.cuda.is_available() else "cpu")

    # Select dynamic/static node types
    config.static_node_types = {'Tissue': (len(adata.obs['Tissue_idx'].unique()), config.d_tissue),
                                'metagenes': (config.meta_G, config.d_gene)}
    config.dynamic_node_types = {'Participant ID': (len(adata.obs['Participant ID'].unique()), config.d_patient)}

    # Model
    config.G = adata.shape[-1]
    model = HypergraphNeuralNet(config).to(device)  # .double()

    # Train
    def rho(x, out):
        x_pred = out['px_rate'].detach().cpu().numpy()
        return np.mean(pearson_correlation_score(x, x_pred, sample_corr=True))
    metric_fns = [rho]
    train(config,
          model=model,
          loader=train_loader,
          val_loader=val_loader,
          device=device,
          preprocess_fn=None,
          compute_metrics_train=False,
          metric_fns=metric_fns)

    torch.save(model.state_dict(), 'data/model.pth') 
