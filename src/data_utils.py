"""
This file contains various utilities to manipulate multi-tissue data
"""

import numpy as np
import torch

# =================================
# AnnData utils
# =================================

def select_obs(adata, obs_dict):
    """
    Select samples satisfying conditions
    :param adata: AnnData object
    :param obs_dict: dictionary: obs_name -> list of accepted values
    :return: filtered adata
    """
    mask = np.full(adata.shape[0], True)
    for k, v in obs_dict.items():
        m = adata.obs[k].isin(v)
        mask = np.logical_and(mask, m)
    return adata[mask]


def filter_not_obs(adata, obs_dict):
    """
    Selects samples in adata such that their obs does not match any obs combination in obs_dict i.e. the i-th
    combination in obs_dict corresponds to obs_dict[k][i] for all k
    :param adata: AnnData with all samples that need to be filtered
    :param obs_dict: Dict {obs_key: obs_values}. obs_dict[k][i] and obs_dict[k'][i] correspond to same sample in filter
    :return: Filtered AnnData where observations do not match any filter in obs_dict
    """
    mask = np.full((adata.shape[0], 1), True)  # Will be broadcasted
    for k, v in obs_dict.items():
        m = np.array(adata.obs[k].values)[:, None] == np.array(obs_dict[k])[None, :]
        mask = np.logical_and(mask, m)
    mask = ~np.any(mask, axis=-1)
    return adata[mask]


def select_overlapping_genes(adata_1, adata_2, var_key='Symbol'):
    """
    Selects set of overlapping genes
    :param adata_1: AnnData 1
    :param adata_2: AnnData 2
    :return: Aligned AnnDatas
    """
    overlapping_genes = np.intersect1d(adata_1.var[var_key].values, adata_2.var[var_key].values)

    # Select overlapping genes between v8 and v9
    gene_mask_1 = [g in overlapping_genes for g in adata_1.var[var_key]]
    adata_1 = adata_1[:, gene_mask_1]
    gene_mask_2 = [g in overlapping_genes for g in adata_2.var[var_key]]
    adata_2 = adata_2[:, gene_mask_2]

    # Align genes by name
    sorted_idxs = np.argsort(adata_1.var[var_key])
    adata_1 = adata_1[:, sorted_idxs]
    sorted_idxs = np.argsort(adata_2.var[var_key])
    adata_2 = adata_2[:, sorted_idxs]

    return adata_1, adata_2


# =================================
# Individual train/test splitting
# =================================

def split_patient_train_test(patients, train_rate=0.8, seed=0):
    """
    Splits individual in two sets
    :param patients: list of individual IDs
    :param train_rate: Proportion of individuals belonging to the first set, the train set.
                       The remaining individuals belong to the second set, the test set.
    :param seed: random seed to ensure that the split are consistent across runs.
    :return: list of train individual IDs, list of test individual IDs
    """
    if seed is not None:
        np.random.seed(seed)
    unique_idxs = np.unique(patients)
    np.random.shuffle(unique_idxs)
    nb_unique = len(unique_idxs)
    split_point = int(train_rate * nb_unique)
    train_idxs = unique_idxs[:split_point]
    test_idxs = unique_idxs[split_point:]
    return train_idxs, test_idxs


# ====================================
# Mapping names to unique integer IDs
# ====================================

def map_to_ids(values, mapping=None):
    """
    Maps list of values to unique IDs (integers from 0 to N-1, extremes included, where N is the
    cardinality of the set of values)
    :param values: list of values to be mapped
    :param mapping: dictionary original value => ID
    :return: mapped values and mapping dictionary
    """
    # values: list of strings
    # mapping: maps (e.g. SUBJID GTEX-1117F) to unique identifiers (integers e.g. 1)
    if mapping is None:
        mapping = {v: i for i, v in enumerate(sorted(np.unique(values)))}
    mapped_values = np.array([mapping[v] for v in values])
    return mapped_values, mapping


# =================================================
# Selecting samples/individuals by tissues
# =================================================

def select_tissues(data, tissues, sampl_ids, selected_tissues):
    """
    Select samples measured in any of the given selected tissues
    :param data: gene expression data. Shape=(nb_samples, nb_genes)
    :param tissues: list indicating the tissue of each sample. Shape=(nb_samples,)
    :param sampl_ids: list with the sample ids. Shape=(nb_samples,)
    :param selected_tissues: list of selected tissues
    :return: updated data, tissues, and sampl_ids to exclusively include samples
             from the selected tissues
    """
    mask = np.isin(tissues, selected_tissues)
    data = data[mask]
    tissues = tissues[mask]
    sampl_ids = sampl_ids[mask]
    print('Selected {} samples'.format(len(sampl_ids)))

    return data, tissues, sampl_ids


def patients_with_tissues_mask(patients, tissues, selected_tissues):
    """
    Returns a mask that selects samples of patients with measured expression values in all selected_tissues
    :param patients: list indicating the patient ID of each sample. Shape=(nb_samples,)
    :param tissues: list indicating the tissue of each sample. Shape=(nb_samples,)
    :param selected_tissues: list of selected tissues
    :return: mask where mask[i] is True if and only if all selected_tissues were measured for patients[i], and False
             otherwise. Shape=(nb_samples,)
    """
    mask = np.zeros_like(patients)
    for pidx in np.unique(patients):
        idxs = np.where(patients == pidx)[0]
        p_tissues = tissues[idxs]
        diff_tissues = np.setdiff1d(selected_tissues, p_tissues)
        if len(diff_tissues) == 0:  # patient has all tissues
            mask[idxs] = 1
    return mask.astype(bool)


# =================================
# Sparsify/densify data in PyTorch
# =================================

def get_hyperedges(expanded_node_map, genes, x=None):
    """
    Flattens the patients, tissue, and gene tensors (matrix indicating the individual, tissue, and genes of each element
    in the dense data). Stacks these indices to form the hyperedge_index. If the data (x) is provided, also flattens it,
    forming the hyperedge attributes.
    :param patients: torch tensor indicating the individual ID of each element. Shape=(nb_samples, nb_genes)
    :param tissues: torch tensor indicating the tissue ID of each element. Shape=(nb_samples, nb_genes)
    :param genes: torch tensor indicating the gene ID of each element. Shape=(nb_samples, nb_genes)
    :param x: Optional. Torch tensor with dense data. Shape=(nb_samples, nb_genes, gene_dim)
    :return: hyperedge indices, shape=(3, nb_hyperedges), and hyperedge attributes, shape=(nb_hyperedges, gene_dim),
             where nb_hyperedges = nb_samples*nb_genes
    """
    nb_samples, nb_genes = genes.shape

    # Obtain hyperedges from idxs (indexing the whole dataset)
    hyperedges = {k: torch.flatten(v) for k, v in expanded_node_map.items()}
    hyperedges['metagenes'] = torch.flatten(genes)

    # Obtain hyperedge attributes
    hyperedge_attr = None
    if x is not None:
        hyperedge_attr = torch.reshape(x, (nb_samples * nb_genes, -1))

    return hyperedges, hyperedge_attr

def sparsify(node_map, genes, x=None):
    """
    Converts dense dataset to sparse dataset (similar to PyTorch Geometric datasets)
    :param patients: torch tensor indicating the individual ID of each sample. Shape=(nb_samples,)
    :param tissues: torch tensor indicating the tissue ID of each sample. Shape=(nb_samples,)
    :param genes: torch tensor indicating the gene ID of each gene. Shape=(nb_genes,)
    :param x: torch tensor with dense data. Shape=(nb_samples, nb_genes, gene_dim)
    :return: sparse dataset consisting of hyperedges and hyperedge_attr:
             - hyperedges: torch tensor indicating the individual, tissue, and gene of each hyperedge. The shape of this
                           tensor is (3, nb_hyperedges), where nb_hyperedges = nb_samples*nb_genes
             - hyperedge_attr: torch tensor with the hyperedge attributes (e.g. flat array of expression values). The
                           shape of this tensor is (nb_hyperedges, gene_dim)
    """
    k = next(iter(node_map))
    nb_samples = node_map[k].shape[0]
    nb_genes = genes.shape[0]

    # Expand tensors to match data shape (nb_samples, nb_genes). Interpretation of the following tensors:
    # - patients_: indicates to which patient each element (i, j) in the data matrix belongs to
    # - tissue_: indicates to which tissue each element (i, j) in the data matrix belongs to
    # - genes_: indicates to which patient each element (i, j) in the data matrix belongs to
    expanded_node_map = {}
    for k, v in node_map.items():
        expanded_node_map[k] = torch.tile(v[:, None], (1, nb_genes))  # Shape=(nb_samples, nb_genes)
    genes_ = torch.tile(genes[None, :], (nb_samples, 1))  # Shape=(nb_samples, nb_genes)

    # Construct hyperedge index
    hyperedges, hyperedge_attr = get_hyperedges(expanded_node_map, genes_, x)

    return hyperedges, hyperedge_attr


def densify(node_map, genes, hyperedge_index, hyperedge_attr):
    """
    Converts sparse dataset (hyperedge_index, hyperedge_attr) into dense dataset that matches the constraints given by
    patients and tissues (for rows, i.e. this function maps the hyperedge values of patients[i] and tissues[i] in the
    i-th row of the dense dataset), and genes (for columns; the hyperedge values of genes[j] are placed in the j-th col
    of the dense dataset). Importantly, this densify function works in a differentiable manner, e.g. gradients will be
    propagated back to the data in sparse format.

    :param patients: torch tensor indicating the individual ID of each sample. Shape=(nb_samples,)
    :param tissues: torch tensor indicating the tissue ID of each sample. Shape=(nb_samples,)
    :param genes: torch tensor indicating the gene ID of each gene. Shape=(nb_genes,)
    :param hyperedge_index: torch tensor indicating the individual, tissue, and gene of each hyperedge. Shape=(3, nb_hyperedges)
    :param hyperedge_attr: torch tensor with the hyperedge attributes (e.g. flat array of expression values).
    :return: tensor with dense dataset. Shape=(nb_samples, nb_genes * gene_dim).
    """
    # Construct dense indices
    k = next(iter(node_map))
    nb_samples = node_map[k].shape[0]
    row_mask = torch.full((nb_samples,), True)
    for k in node_map.keys():
        row_mask = row_mask.to(node_map[k].device) & (hyperedge_index[k][:, None] == node_map[k])
    # Shape row_mask = (nb_hyperedges=nb_samples*nb_metagenes, nb_samples)
    col_mask = hyperedge_index['metagenes'][:, None] == genes
    row = torch.where(row_mask)[1]
    col = torch.where(col_mask)[1]

    # Fill in output tensor
    out_shape = (nb_samples, genes.shape[0], hyperedge_attr.shape[-1])
    out = hyperedge_attr.new_full(out_shape, 0)  # Important: default value for requires_grad is requires_grad=True
    out[row, col] = hyperedge_attr
    out = torch.reshape(out, (-1, genes.shape[0] * hyperedge_attr.shape[-1]))

    return out