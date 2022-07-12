"""
This file defines the Pytorch MultiTissueDataset
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from src.data import Data
from src.data_utils import select_obs, filter_not_obs


class HypergraphDataset(Dataset):
    """
    This MultiTissueDataset takes care of masking out individuals that do not have the specified source/target tissues
    as well as splitting data sampling the gene expression measurements from individuals according to the source/target
    tissues
    """

    def __init__(self, adata, adata_target=None, obs_source=None, obs_target=None, donor_key='Participant ID',
                 static=False, disjoint=False, verbose=False, dtype=torch.float32):  # max_samples=500
        """
        :param adata: torch tensor with dense data. Shape=(nb_samples, nb_genes, gene_dim)
        :param static: Whether to fix the source and target tissues of each individual. In other words, querying the
               same individual many times will yield the same source and target tissues.
        :param verbose: Whether to print details.
        """
        self.dtype = dtype
        self.dtype = dtype
        self.static = static
        self.disjoint = disjoint
        # self.max_samples = max_samples  # Maximum samples per donor

        # Select samples that satisfy source and target obs
        self.obs_source, self.obs_target = obs_source, obs_target
        adata_source = adata

        if adata_target is None:
            adata_target = adata

        if obs_source is not None:
            adata_source = select_obs(adata_source, obs_source)
        if obs_target is not None:
            adata_target = select_obs(adata_target, obs_target)

        # Discard source/target samples without matching donors
        self.donor_key = donor_key
        source_donors = adata_source.obs[donor_key].unique()
        target_donors = adata_target.obs[donor_key].unique()
        donor_ids = np.intersect1d(source_donors, target_donors)
        self.nb_donors = len(donor_ids)
        self.adata_source = select_obs(adata_source, {donor_key: donor_ids})
        self.adata_target = select_obs(adata_target, {donor_key: donor_ids})
        if verbose:
            print(f'Selected {self.adata_source.shape[0]} source and {self.adata_target.shape[0]} target samples of {self.nb_donors} unique donors')

        # Create patient map giving a unique ID from [0, nb_patients) to each individual
        self.donor_map = {i: v for i, v in enumerate(sorted(donor_ids))}
        self.donor_map_inv = {v: i for i, v in self.donor_map.items()}

        # Fix source/target tissues of each individual
        if static:
            self.donor_adata_source = []
            self.donor_adata_target = []
            for i, pidx in enumerate(sorted(donor_ids)):
                donor_adata_source, donor_adata_target = self._get_source_target(i, static=False)
                self.donor_adata_source.append(donor_adata_source)
                self.donor_adata_target.append(donor_adata_target)

    def __len__(self):
        """
        :return: number of individuals in the dataset
        """
        return self.nb_donors

    def _get_source_target(self, i, static=False):
        """
        Returns data for the i-th individual in the dataset
        :param i: index in [0, nb_patients)
        :param static: whether to return the static source/target indices of the samples of the i-th individual
        :return: source/target indices (indexing rows in self.data) of the samples for the i-th individual
        """
        if static:
            return self.donor_adata_source[i], self.donor_adata_target[i]

        # Find actual donor index
        didx = self.donor_map[i]

        # Select samples
        donor_adata_source = select_obs(self.adata_source, {self.donor_key: [didx]})
        donor_adata_target = select_obs(self.adata_target, {self.donor_key: [didx]})

        if self.disjoint:  # Source and target sets do not contain same samples
            # print(didx, donor_adata_source.shape[0])
            n = donor_adata_source.shape[0]
            nb_source = 1 + np.random.choice(n-1, 1)  # Number of source samples. Min: 1. Max: n - 1
            idxs = np.random.choice(n, nb_source, replace=False)
            source_obs = {k: donor_adata_source.obs[k].values[idxs] for k in donor_adata_source.obs.columns}
            donor_adata_source = select_obs(donor_adata_source, source_obs)
            donor_adata_target = filter_not_obs(donor_adata_target, source_obs)

        return donor_adata_source, donor_adata_target

    def __getitem__(self, i):
        """
        Return source/target samples for i-th individual
        :param i: Index in [0, nb_donors)
        :return: x_source, x_target, tissues_source, tissues_target, patients_source, patients_target
        """
        if torch.is_tensor(i):
            i = i.tolist()

        # Find sample indices in data matrix corresponding to patient pidx
        donor_adata_source, donor_adata_target = self._get_source_target(i, static=self.static)

        return Data(donor_adata_source, donor_adata_target, dtype=self.dtype)