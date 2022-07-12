import scanpy as sc
import numpy as np
from collections import Counter
from src.data_utils import *
from src.train_utils import *
from src.distributions import *
import torch

# =================================
# Load GTEx v9 data
# =================================
GTEX_v9_FILE = '/local/scratch-2/rv340/GTEx/v9/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad'


def GTEx_v9_adata(file=GTEX_v9_FILE):
    adata = sc.read(file)
    adata.var['Symbol'] = adata.var['Approved symbol']

    return adata


def mask_differentially_expressed_genes(adata, ct_key='Broad cell type', min_counts=50, pval_threshold=0.05):
    adata_ = adata.copy()

    # Filter cells
    sc.pp.filter_cells(adata_, min_counts=min_counts)

    # Total-count normalize (library-size correct) the data matrix to 10,000 reads per cell, so that counts become comparable among cells.
    adata_.X = adata_.layers['counts']
    sc.pp.normalize_total(adata_, inplace=True, target_sum=1e4)

    # Logarithmise data
    sc.pp.log1p(adata_)

    # Statistical test
    sc.tl.rank_genes_groups(adata_, ct_key, use_raw=False, method='wilcoxon', key_added='wilcoxon')

    # Select genes with adjusted pval < threshold
    mask_selected = np.zeros(adata_.shape[1])
    for ct in adata_.obs[ct_key].unique():
        pvals = adata_.uns['wilcoxon']['pvals_adj'][ct]
        mask_selected += (pvals < pval_threshold)
    mask_selected = mask_selected >= 1
    return mask_selected


def select_genes(adata_v8=None, adata_v9=None, strategy='highly variable v8', n_top_genes=3000, pval_threshold=0.05):
    # Select highly variable genes
    if strategy == 'highly variable v8':
        assert adata_v8 is not None
        sc.pp.highly_variable_genes(adata_v8, n_top_genes=n_top_genes, flavor='seurat_v3', inplace=True)
        if adata_v9 is not None:
            adata_v9 = adata_v9[:, adata_v8.var.highly_variable]
        adata_v8 = adata_v8[:, adata_v8.var.highly_variable]
    elif strategy == 'highly variable v9':
        assert adata_v9 is not None
        sc.pp.highly_variable_genes(adata_v9, n_top_genes=n_top_genes, flavor='seurat_v3', inplace=True)
        if adata_v8 is not None:
            adata_v8 = adata_v8[:, adata_v9.var.highly_variable]
        adata_v9 = adata_v9[:, adata_v9.var.highly_variable]
    elif strategy == 'differentially expressed':
        assert adata_v9 is not None
        mask = mask_differentially_expressed_genes(adata_v9, pval_threshold=pval_threshold)
        if adata_v8 is not None:
            adata_v8 = adata_v8[:, mask]
        adata_v9 = adata_v9[:, mask]
    else:
        raise ValueError('Unknown strategy {}'.format(strategy))
    return adata_v8, adata_v9


# =================================
# Calculate GTEx v9 signatures
# =================================

def GTEx_v9_signatures(adata_v8, adata_v9, ct_key='Broad cell type', threshold=10):
    ct_adatas = []
    for donor_id in adata_v9.obs['Participant ID'].unique():
        donor_adata = adata_v9[adata_v9.obs['Participant ID'] == donor_id]
        for tissue in donor_adata.obs['Tissue'].unique():
            donor_tissue_adata = donor_adata[donor_adata.obs['Tissue'] == tissue]
            for ct in donor_tissue_adata.obs[ct_key].unique():
                donor_tissue_ct_adata = donor_tissue_adata[donor_tissue_adata.obs[ct_key] == ct]

                # Aggregate
                ct_counts = donor_tissue_ct_adata.layers['counts'].toarray().sum(axis=0)[None, :]
                aggr_adata = sc.AnnData(ct_counts)
                aggr_adata.obs['Participant ID'] = donor_id
                aggr_adata.obs['Tissue'] = tissue
                aggr_adata.obs[ct_key] = ct
                aggr_adata.obs['n_cells'] = donor_tissue_ct_adata.shape[0]
                aggr_adata.obs['Sex'] = donor_tissue_ct_adata.obs['Sex'][0]
                aggr_adata.obs['Age_bin'] = donor_tissue_ct_adata.obs['Age_bin'][0]

                ct_adatas.append(aggr_adata)

    ct_adata_v9 = sc.concat(ct_adatas)
    ct_adata_v9.obs.index = np.arange(ct_adata_v9.shape[0])
    ct_adata_v9.obs.index = ct_adata_v9.obs.index.astype(str)

    # Prepare adata for hypergraph dataset
    ct_adata_v9.layers['x'] = ct_adata_v9.X
    ct_adata_v9.obs['Participant ID_dyn'] = ct_adata_v9.obs['Participant ID']
    ct_adata_v9.obs['Age'] = ct_adata_v9.obs['Age_bin']
    if 'Age_dict' in adata_v8.uns:
        donor_age = ct_adata_v9.obs['Age'].map(adata_v8.uns['Age_dict'])
    else:
        donor_age = ct_adata_v9.obs['Age']
    donor_sex = ct_adata_v9.obs['Sex'].map(adata_v8.uns['Sex_dict'])
    ct_adata_v9.obsm['Participant ID_feat'] = np.stack((donor_age, donor_sex), axis=-1)
    ct_adata_v9.obs['n_cells_misc'] = ct_adata_v9.obs['n_cells']

    # Make tissue indices homogeneous
    ct_adata_v9.obs['Tissue'].unique()
    tissue_dict_v9 = {'Skeletal muscle': 33,
                      'Esophagus muscularis': 26,
                      'Lung': 31,
                      'Prostate': 38,
                      'Skin': 40,  # Assuming sun exposed. Sun not exposed is 39
                      'Heart': 27,  # Assuming heart attrial. Heart left ventricle is 28
                      'Esophagus mucosa': 25,
                      'Breast': 19
                      }
    ct_adata_v9.uns['Tissue_dict'] = tissue_dict_v9
    ct_adata_v9.obs['Tissue_idx'] = ct_adata_v9.obs['Tissue'].map(tissue_dict_v9)

    # Discard underrepresented cell-types
    # TODO: Instead of discarding underrepresented cell-types, once could generate several profiles per individual-tissue-ct
    # by selecting a random subset of cell-types. This might allow to better capture the variation of each combination.
    selected_ct = {k: v for k, v in Counter(ct_adata_v9.obs[ct_key].values).items() if v >= threshold}
    ct_adata_v9 = select_obs(ct_adata_v9, {ct_key: selected_ct.keys()})

    # Map to indices
    ct_adata_v9.obs['Cell type_idx'], ct_dict = map_to_ids(ct_adata_v9.obs[ct_key].values)
    ct_adata_v9.uns['ct_dict'] = ct_dict

    # Set layers
    ct_adata_v9.layers['x'] = ct_adata_v9.X

    return ct_adata_v9


# =================================
# Infer signatures
# =================================

def infer_signatures(d, model, device, inference_mode='mean', generative_mode='sample', **kwargs):
    model.eval()
    with torch.no_grad():
        d = d.to(device)
        node_features = encode(d, model, **kwargs)

        # Get latent variables
        (dynamic_node_features_, static_node_features) = node_features

        if inference_mode == 'sample':
            sample = torch.distributions.normal.Normal(loc=dynamic_node_features_['Participant ID']['mu'],
                                                       scale=dynamic_node_features_['Participant ID']['var']).sample()
            dynamic_node_features_['Participant ID']['latent'] = sample
        elif inference_mode == 'mean':
            # Set latents to mean
            dynamic_node_features_['Participant ID']['latent'] = dynamic_node_features_['Participant ID']['mu']
        else:
            raise ValueError('Inference mode {} not understood'.format(generative_mode))
        node_features_ = (dynamic_node_features_, static_node_features)

        # Compute signatures
        out = decode(d, model, node_features_, **kwargs)
        dist = ZeroInflatedNegativeBinomial(mu=out['px_rate'], theta=out['px_r'],
                                            zi_logits=out['px_dropout'])

        # Sample from generative model
        if generative_mode == 'sample':
            x_pred = dist.sample()
        elif generative_mode == 'mean':
            x_pred = dist.mean
        elif generative_mode == 'mu':
            x_pred = out['px_rate']
        elif generative_mode == 'dropout':
            x_pred = torch.exp(out['px_dropout']) / (1 + torch.exp(out['px_dropout']))
        else:
            raise ValueError('Generative mode {} not understood'.format(generative_mode))

    inferred_signatures = x_pred.cpu().numpy()
    return inferred_signatures


# =================================
# Convolution
# =================================

def convolve(adata_v9, participant_id, target_tissue, target_cts, ct_key='Broad cell type'):
    # Select cells with matching criteria
    obs_mask = (adata_v9.obs['Participant ID'] == participant_id) * \
               (adata_v9.obs['Tissue'] == target_tissue) * \
               (adata_v9.obs[ct_key].isin(target_cts))
    selected_adata = adata_v9[obs_mask]

    unique_cts = list(selected_adata.obs[ct_key].unique())
    # cell_type_counts = {k: v for k, v in Counter(selected_adata.obs[ct_key]).items()}
    cell_type_counts = {ct: (selected_adata.obs[ct_key] == ct).sum() for ct in unique_cts}
    total_cells = sum(list(cell_type_counts.values()))
    cell_type_proportions = {k: v / total_cells for k, v in cell_type_counts.items()}

    # Aggregate
    X_convolved = selected_adata.X.toarray().sum(axis=0)

    return X_convolved, cell_type_proportions, total_cells


# =================================
# Transfer learning utilities
# =================================

def ct_predict_v1(self, target_hyperedge_index, node_features, **kwargs):
    """
    Given the latent features of all nodes in the hypergraph, predicts the metagene values (i.e. hyperedge attributes)
    of all hyperedges in target_hyperedge_index
    :param target_hyperedge_index: indices of the indices of the hyperedges (similar to edge_index of
           PyTorch Geometric) whose hyperedge values need to be predicted. Shape=(3, nb_hyperedges)
    :param node_features: tuple of node features (individual features, tissue features, metagene features), where:
            - patient_features: features of individual nodes. Shape=(nb_patients, d_patient)
            - tissue_features: features of tissue nodes. Shape=(nb_tissues, d_tissue)
            - metagene_features: features of tissue nodes. Shape=(nb_metagenes, d_metagene)
    :return: torch tensor with predicted hyperedge features for each hyperedge in target_hyperedge_index.
            Shape=(nb_hyperedges, d_edge_attr).
    """
    dynamic_node_features, static_node_features = node_features

    # Get sampled latent values
    dynamic_node_features_ = {k: v['latent'] for k, v in dynamic_node_features.items()}

    # Predict metagene values
    node_features_ = {**dynamic_node_features_, **static_node_features}

    # Construct modified node features
    catted_features = []
    for k in sorted(node_features_.keys()):
        feat = node_features_[k][target_hyperedge_index[k]]

        if k == 'Tissue':  # Modify tissue features with cell-type features
            feat += static_node_features['Cell type'][target_hyperedge_index['Cell type']]
        elif k == 'Cell type':
            # do nothing, continue to next iteration
            continue

        catted_features.append(feat)

    catted_features = torch.cat(catted_features, dim=-1)

    return self.prediction_mlp(catted_features)


def ct_predict_v2(self, target_hyperedge_index, node_features, **kwargs):
    """
    Given the latent features of all nodes in the hypergraph, predicts the metagene values (i.e. hyperedge attributes)
    of all hyperedges in target_hyperedge_index
    :param target_hyperedge_index: indices of the indices of the hyperedges (similar to edge_index of
           PyTorch Geometric) whose hyperedge values need to be predicted. Shape=(3, nb_hyperedges)
    :param node_features: tuple of node features (individual features, tissue features, metagene features), where:
            - patient_features: features of individual nodes. Shape=(nb_patients, d_patient)
            - tissue_features: features of tissue nodes. Shape=(nb_tissues, d_tissue)
            - metagene_features: features of tissue nodes. Shape=(nb_metagenes, d_metagene)
    :return: torch tensor with predicted hyperedge features for each hyperedge in target_hyperedge_index.
            Shape=(nb_hyperedges, d_edge_attr).
    """
    dynamic_node_features, static_node_features = node_features

    # Get sampled latent values
    dynamic_node_features_ = {k: v['latent'] for k, v in dynamic_node_features.items()}

    # Predict metagene values
    node_features_ = {**dynamic_node_features_, **static_node_features}

    # Construct modified node features
    catted_features = []
    for k in sorted(node_features_.keys()):
        feat = node_features_[k][target_hyperedge_index[k]]
        catted_features.append(feat)

    catted_features = torch.cat(catted_features, dim=-1)

    return self.prediction_mlp(catted_features)
