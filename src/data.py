"""
Defines Data class encapsulating data source/target gene expression data
"""

import torch
from src.data_utils import map_to_ids
import numpy as np


class Data:
    """
    Class encapsulating instances of source tissue and target tissue gene expression for one or multiple individuals
    """

    def __init__(self, donor_adata_source=None, donor_adata_target=None, dtype=torch.float32):
        """
        All obs in data will be used. Please discard any unused obs outside of this class and make sure
        that they have the appropriate type (so that they can be converted to PyTorch tensors)
        """
        self.dtype = dtype
        self.source = {}
        self.target = {}
        self.source_misc = {}
        self.target_misc = {}  # Storing number of cells here (for deconvoltion experiment)
        self.source_features = {}
        self.target_features = {}
        self.source_dynamic = {}
        self.target_dynamic = {}

        if donor_adata_source is not None:
            for k in donor_adata_source.obs.columns:  # Static keys
                v = donor_adata_source.obs[k]
                if k.endswith('_idx'):
                    self.source[k.replace('_idx', '')] = torch.tensor(v)
                elif k.endswith('_dyn'):  # Dynamic IDs
                    self.source_dynamic[k.replace('_dyn', '')] = v
                elif k.endswith('_misc'):
                    self.source_misc[k.replace('_misc', '')] = torch.tensor(v)

            for k, v in donor_adata_source.obsm.items():
                if k.endswith('_feat'):
                    self.source_features[k.replace('_feat', '')] = torch.tensor(v)

            self.x_source = torch.tensor(donor_adata_source.layers['x'].toarray(), dtype=dtype)

        if donor_adata_target is not None:
            for k in donor_adata_target.obs.columns:  # Static keys
                v = donor_adata_target.obs[k]
                if k.endswith('_idx'):
                    self.target[k.replace('_idx', '')] = torch.tensor(v)
                elif k.endswith('_dyn'):  # Dynamic IDs
                    self.target_dynamic[k.replace('_dyn', '')] = v
                elif k.endswith('_misc'):
                    self.target_misc[k.replace('_misc', '')] = torch.tensor(v)

            for k, v in donor_adata_target.obsm.items():
                if k.endswith('_feat'):
                    self.target_features[k.replace('_feat', '')] = torch.tensor(v)

            self.x_target = torch.tensor(donor_adata_target.layers['x'].toarray(), dtype=dtype)

        # Store cell IDs
        self.map_dynamic_idxs()  # creates unique cell identifiers

    def map_dynamic_idxs(self):
        """
        Map dynamic elements to unique indices
        E.g. Creates unique identifiers for each cell in [0, nb_cells_in_data)
        """
        def increasing_index_map(values, features):
            map = {}
            out_indices = np.zeros_like(values, dtype=int)
            n_unique = len(np.unique(values))
            out_features = np.zeros((n_unique, features.shape[-1]))
            for i, v in enumerate(values):
                if v in map:
                    out_indices[i] = map[v]
                else:
                    idx = len(map)
                    out_indices[i] = idx
                    out_features[idx] = features[i]  # Create features such that they return the appropriate value when
                                                     # indexed by the dynamic idx
                    map[v] = idx
            return out_indices, out_features, map

        self.node_features = {}
        # TODO: Sanity check -- there cannot be any target nodes that do not belong to source nodes
        # This might cause issues in aggregation operations, i.e. they have no incoming messages
        for k in self.source_dynamic.keys():  # assumes same set of keys for source and target
            v_source = self.source_dynamic[k]
            v_target = self.target_dynamic[k]
            v = np.concatenate((v_source, v_target))

            features = np.concatenate((self.source_features[k], self.target_features[k]))
            v_idxs, out_features, v_map = increasing_index_map(v, features)  # map_to_ids(v)
            self.source[k] = torch.tensor(v_idxs[:v_source.shape[0]])
            self.target[k] = torch.tensor(v_idxs[v_source.shape[0]:])

            # Concatenate features as they are shared between source and target
            # sf = self.source_features[k]
            # print('sf shape', sf.shape)
            # print('v_idxs shape', v_idxs.shape)
            # print('sfi shape', sf[v_idxs])
            # tf = self.target_features[k]
            # catted_f = torch.cat([sf, tf], dim=0)
            # self.source_features[k] = torch.tensor(out_features)  # catted_f
            # self.target_features[k] = torch.tensor(out_features)  # catted_f
            self.node_features[k] = torch.tensor(out_features, dtype=self.dtype)

    @staticmethod
    def from_datalist(datalist):
        """
        Creates a single Data object for a list of Data objects
        :param datalist: list of Data objects
        :return: merged Data object
        """
        data = Data()

        # Store static features
        for k in datalist[0].source.keys():
            data.source[k] = torch.cat([d.source[k] for d in datalist], dim=0)
        for k in datalist[0].target.keys():
            data.target[k] = torch.cat([d.target[k] for d in datalist], dim=0)

        data.x_source = torch.cat([d.x_source for d in datalist], dim=0)
        data.x_target = torch.cat([d.x_target for d in datalist], dim=0)

        # Store dynamic features
        for k in datalist[0].source_features.keys():
            data.source_features[k] = torch.cat([d.source_features[k] for d in datalist], dim=0)
        for k in datalist[0].target_features.keys():
            data.target_features[k] = torch.cat([d.target_features[k] for d in datalist], dim=0)

        # Store miscellaneous features
        for k in datalist[0].source_misc.keys():
            data.source_misc[k] = torch.cat([d.source_misc[k] for d in datalist], dim=0)
        for k in datalist[0].target_misc.keys():
            data.target_misc[k] = torch.cat([d.target_misc[k] for d in datalist], dim=0)

        # Store dynamic IDs
        for k in datalist[0].source_dynamic.keys():
            data.source_dynamic[k] = np.concatenate([d.source_dynamic[k] for d in datalist])
        for k in datalist[0].target_dynamic.keys():
            data.target_dynamic[k] = np.concatenate([d.target_dynamic[k] for d in datalist])
        data.map_dynamic_idxs()

        return data

    def to(self, device):
        """
        Maps data to device
        :param device: device on to which the data will be mapped
        """
        for k, v in self.source.items():
            self.source[k] = v.to(device)
        for k, v in self.source_misc.items():
            self.source_misc[k] = v.to(device)
        for k, v in self.target.items():
            self.target[k] = v.to(device)
        for k, v in self.target_misc.items():
            self.target_misc[k] = v.to(device)

        self.x_source = self.x_source.to(device)
        self.x_target = self.x_target.to(device)

        for k, v in self.node_features.items():
            self.node_features[k] = v.to(device)

        return self