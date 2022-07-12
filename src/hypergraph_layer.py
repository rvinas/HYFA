"""
Defines hypergraph layer
"""

import torch
import torch.nn as nn
from src.hnn_utils import *
from src.metagene_encoders import *
from src.metagene_decoders import *


class GATHypergraphLayer(torch.nn.Module):

    def __init__(self, dynamic_node_types, static_node_types, d_edge_attr=1, d_edge=50, n_hidden_layers=2, n_heads=5, dropout=0.1, norm='batch', activation='relu', update_edge_attr=False, attention_strategy='patient'):
        """
        :param d_patient: dimension of individual node features
        :param d_gene: dimension of metagene node features
        :param d_tissue: dimension of tissue node features
        :param d_edge_attr: dimension of hyperedge attributes
        :param d_edge: dimension of messages
        :param n_hidden_layers: Number of hidden layers for the MLPs
        :param n_heads: Number of heads of the attention-based aggregation mechanism
        :param dropout: Dropout probability value
        :param norm: whether to apply batch/layer normalisation
        :param activation: activation function
        :param update_edge_attr: whether to update the hyperedge attributes at each graph layer
        :param attention_strategy: strategy for computing the attention coefficients:
               - 'patient': Computes softmax over all incoming messages to patient
               - 'patient_metagene': Computes softmax for each metagene over all source tissues
               - 'patient_tissue': Computes softmax for each source tissue over all metagenes
        """
        super().__init__()
        self.n_heads = n_heads
        self.update_edge_attr = update_edge_attr
        self.attention_strategy = attention_strategy

        # Compute dynamic/static feature dims
        dynamic_feature_dim = 0
        static_feature_dim = 0
        for k, v in dynamic_node_types.items():
            n, dim = v
            dynamic_feature_dim += dim
        for k, v in static_node_types.items():
            n, dim = v
            static_feature_dim += dim
        total_feature_dim = static_feature_dim + dynamic_feature_dim

        # Edge MLPs
        self.edge_mlp = MLP(in_dim=total_feature_dim + d_edge_attr,
                            out_dim=n_heads * d_edge,
                            n_layers=n_hidden_layers,
                            dropout=dropout,
                            norm=norm,
                            activation=activation)

        # Aggregator functions. Note: Currently only sum used as messages are currently weighted by attention scores
        self.aggregators = ['sum']  # ['sum', 'max', 'min', 'mean', 'std']

        # TODO: Attention aggregation
        hdim = static_feature_dim
        self.att_agg_mlp = nn.Sequential(nn.Linear(static_feature_dim, hdim, bias=False), nn.LeakyReLU(0.1),
                                         nn.Linear(hdim, n_heads, bias=False))

        # Dynamic node MLPs
        self.dynamic_node_mlp = nn.ModuleDict()
        for k, v in dynamic_node_types.items():
            n, dim = v
            self.dynamic_node_mlp[k] = MLP(in_dim=dim + len(self.aggregators) * n_heads * d_edge,
                                           out_dim=dim,
                                           n_layers=n_hidden_layers,
                                           dropout=dropout,
                                           norm=norm,
                                           activation=activation)

        # Hyperedge attr update MLPs
        if update_edge_attr:
            self.hyperedge_attr_mlp = MLP(in_dim=total_feature_dim + d_edge_attr,
                                          out_dim=d_edge_attr,
                                          n_layers=n_hidden_layers,
                                          dropout=dropout,
                                          norm=norm,
                                          activation=activation)

    def forward(self, hyperedge_index, hyperedge_attr, node_features_per_node):
        """
        Performs message passing:
            1) Computes messages for each (individual, tissue, and metagene) in hyperedge_index
            2) Aggregates messages with attention-based aggregation mechanism for each individual
            3) Computes updates for each node (currently individual nodes only)

        :param hyperedge_index: indices of hyperedges (similar to edge_index of PyTorch Geometric). Shape=(3, nb_hyperedges)
        :param hyperedge_attr: hyperedge features (similar to edge_attr of PyTorch Geometric). Shape=(nb_hyperedges, d_edge_attr)
        :param node_features_per_node: tuple with *all* node features (individual_features, tissue_features, metagene_features):
                                       - individual_features: (nb_patients, d_patient)
                                       - tissue_features: (nb_tissues, d_tissue)
                                       - metagene_features: (nb_metagenes, d_metagene)
        :return: Node feature updates (tuple) and hyperedge attributes
        """
        dynamic_node_features, static_node_features = node_features_per_node

        # Expanded node features
        dynamic_node_features_e = {}
        static_node_features_e = {}
        for k, v in hyperedge_index.items():
            if k in dynamic_node_features:
                dynamic_node_features_e[k] = dynamic_node_features[k][v]
            elif k in static_node_features:
                static_node_features_e[k] = static_node_features[k][v]
        node_features_e = {**dynamic_node_features_e, **static_node_features_e}

        # Message passing
        m = self.messages(hyperedge_attr, node_features_e)
        m_updates = self.aggregate(m, hyperedge_index, dynamic_node_features_e, static_node_features_e, dynamic_node_features)
        dynamic_node_features = self.update(dynamic_node_features, m_updates)

        # Update hyperedge attributes
        if self.update_edge_attr:
            hyperedge_attr_updates = self.hyperedge_attr_update(hyperedge_attr, node_features_e)
            hyperedge_attr = hyperedge_attr + hyperedge_attr_updates

        # Return updates per node
        return dynamic_node_features, hyperedge_attr

    def messages(self, hyperedge_attr, node_features_e):
        """
        Computes individual messages
        :param hyperedge_attr: hyperedge features (similar to edge_attr of PyTorch Geometric). Shape=(nb_hyperedges, d_edge_attr)
        :param node_features_e: features of all hyperedges. Tuple (patient_features, tissue_features, gene_features) where:
                              - patient_features: patient features of each hyperedge. Shape=(nb_hyperedges, d_patient)
                              - tissue_features: tissue features of each hyperedge. Shape=(nb_hyperedges, d_tissue)
                              - gene_features: patient features of each hyperedge. Shape=(nb_hyperedges, d_metagene)
        :return: Torch tensor with messages to individuals. Shape=(nb_hyperedges, message_dim)
        """
        # Compute messages
        # print({k: node_features_e[k].shape for k in node_features_e.keys()})
        # print({k: node_features_e[k].get_device() for k in node_features_e.keys()})
        catted_features = torch.cat([node_features_e[k] for k in sorted(node_features_e.keys())]
                                    + [hyperedge_attr], dim=-1)
        messages = self.edge_mlp(catted_features)
        return messages

    def aggregate(self, messages, hyperedge_index, dynamic_node_features_e, static_node_features_e, dynamic_node_features):
        """
        Aggregates all the messages sent to the same individual
        :param messages: torch tensor with shape (nb_hyperedges, message_dim)
        :param hyperedge_index: indices of hyperedges (similar to edge_index of PyTorch Geometric). Shape=(3, nb_hyperedges)
        :param node_features: features of all hyperedges. Tuple (patient_features, tissue_features, gene_features) where:
                      - patient_features: patient features of each hyperedge. Shape=(nb_hyperedges, d_patient)
                      - tissue_features: tissue features of each hyperedge. Shape=(nb_hyperedges, d_tissue)
                      - gene_features: patient features of each hyperedge. Shape=(nb_hyperedges, d_metagene)
        :return: Aggregated messages. Shape=(nb_individuals, message_dim)
        """
        # Compute attention coefficients based on static node features
        catted_static_features = torch.cat([static_node_features_e[k] for k in sorted(static_node_features_e.keys())], dim=-1)
        e = self.att_agg_mlp(catted_static_features)  # Shape=(n_messages, n_heads)

        messages = torch.reshape(messages, (messages.shape[0], self.n_heads, -1))

        aggregated_messages = {}
        for k in dynamic_node_features_e.keys():
            # Get attention coefficients for node type k
            idxs = hyperedge_index[k]  # Softmax over all incoming messages
            softmax_idxs = idxs
            # softmax_idxs = hyperedge_index['Tissue']
            # softmax_idxs = unique_ids(hyperedge_index['Tissue'], hyperedge_index['Participant ID'])  # Unique ids for each patient and tissue
            att_coeffs = scatter_softmax(e, softmax_idxs)  # Shape=(n_messages, n_heads)

            # Weight by att_coeffs and concatenate across heads
            m = att_coeffs[..., None] * messages  # Shape=(n_messages, n_heads, d_edge)
            m = torch.reshape(m, (m.shape[0], -1))  # Shape=(n_messages, n_heads * d_edge)

            # Aggregate messages
            m = message_aggregation(messages=m,
                                    idxs=idxs,
                                    dim_size=dynamic_node_features[k].shape[0],  # Number of nodes of original tensor
                                    aggregators=self.aggregators)
            aggregated_messages[k] = m

        return aggregated_messages

    def update(self, dynamic_node_features, message_updates):
        """
        Computes node feature updates (currently individuals only).
        TODO: Update node features directly?
        :param node_features_per_node: tuple with *all* node features (individual_features, tissue_features, metagene_features):
                                       - individual_features: (nb_patients, d_patient)
                                       - tissue_features: (nb_tissues, d_tissue)
                                       - metagene_features: (nb_metagenes, d_metagene)
        :param message_updates: Aggregated messages. Shape=(nb_individuals, message_dim)
        :return: Node feature updates (currently individuals only). Shape=(nb_individuals, d_patient)
        """
        for k in dynamic_node_features.keys():
            h = torch.cat((dynamic_node_features[k], message_updates[k]), dim=-1)
            dynamic_node_features[k] = self.dynamic_node_mlp[k](h)

        return dynamic_node_features

    def hyperedge_attr_update(self, hyperedge_attr, node_features_e):
        """
        Updates hyperedge attributes
        :param hyperedge_attr: hyperedge features (similar to edge_attr of PyTorch Geometric). Shape=(nb_hyperedges, d_edge_attr)
        :param node_features: features of all hyperedges. Tuple (patient_features, tissue_features, gene_features) where:
                      - patient_features: patient features of each hyperedge. Shape=(nb_hyperedges, d_patient)
                      - tissue_features: tissue features of each hyperedge. Shape=(nb_hyperedges, d_tissue)
                      - gene_features: patient features of each hyperedge. Shape=(nb_hyperedges, d_metagene)
        :return: updated hyperedge attributes. Shape=(nb_hyperedges, d_edge_attr)
        """
        assert self.update_edge_attr

        # Compute messages
        catted_features = torch.cat([node_features_e[k] for k in sorted(node_features_e.keys())]
                                    + [hyperedge_attr], dim=-1)
        hyperedge_attr_updates = self.hyperedge_attr_mlp(catted_features)

        return hyperedge_attr_updates


class MPNNHypergraphLayer(torch.nn.Module):

    def __init__(self, dynamic_node_types, static_node_types, d_edge_attr=1, d_edge=50, n_hidden_layers=2, n_heads=5, dropout=0.1, norm='batch', activation='relu', update_edge_attr=False, attention_strategy='patient'):
        """
        :param d_patient: dimension of individual node features
        :param d_gene: dimension of metagene node features
        :param d_tissue: dimension of tissue node features
        :param d_edge_attr: dimension of hyperedge attributes
        :param d_edge: dimension of messages
        :param n_hidden_layers: Number of hidden layers for the MLPs
        :param n_heads: Number of heads of the attention-based aggregation mechanism
        :param dropout: Dropout probability value
        :param norm: whether to apply batch/layer normalisation
        :param activation: activation function
        :param update_edge_attr: whether to update the hyperedge attributes at each graph layer
        :param attention_strategy: strategy for computing the attention coefficients:
               - 'patient': Computes softmax over all incoming messages to patient
               - 'patient_metagene': Computes softmax for each metagene over all source tissues
               - 'patient_tissue': Computes softmax for each source tissue over all metagenes
       
        Note: Some params are unused, e.g. n_heads.
        """
        super().__init__()
        self.update_edge_attr = update_edge_attr
        
        # Compute dynamic/static feature dims
        dynamic_feature_dim = 0
        static_feature_dim = 0
        for k, v in dynamic_node_types.items():
            n, dim = v
            dynamic_feature_dim += dim
        for k, v in static_node_types.items():
            n, dim = v
            static_feature_dim += dim
        total_feature_dim = static_feature_dim + dynamic_feature_dim

        # Edge MLPs
        self.edge_mlp = MLP(in_dim=total_feature_dim + d_edge_attr,
                            out_dim=d_edge,
                            n_layers=n_hidden_layers,
                            dropout=dropout,
                            norm=norm,
                            activation=activation)

        # Aggregator functions.
        self.aggregators = ['mean']  # ['sum', 'max', 'min', 'mean', 'std']

        # Dynamic node MLPs
        self.dynamic_node_mlp = nn.ModuleDict()
        for k, v in dynamic_node_types.items():
            n, dim = v
            self.dynamic_node_mlp[k] = MLP(in_dim=dim + len(self.aggregators) * d_edge,
                                           out_dim=dim,
                                           n_layers=n_hidden_layers,
                                           dropout=dropout,
                                           norm=norm,
                                           activation=activation)

        # Hyperedge attr update MLPs
        if update_edge_attr:
            self.hyperedge_attr_mlp = MLP(in_dim=total_feature_dim + d_edge_attr,
                                          out_dim=d_edge_attr,
                                          n_layers=n_hidden_layers,
                                          dropout=dropout,
                                          norm=norm,
                                          activation=activation)

    def forward(self, hyperedge_index, hyperedge_attr, node_features_per_node):
        """
        Performs message passing:
            1) Computes messages for each (individual, tissue, and metagene) in hyperedge_index
            2) Aggregates messages with attention-based aggregation mechanism for each individual
            3) Computes updates for each node (currently individual nodes only)

        :param hyperedge_index: indices of hyperedges (similar to edge_index of PyTorch Geometric). Shape=(3, nb_hyperedges)
        :param hyperedge_attr: hyperedge features (similar to edge_attr of PyTorch Geometric). Shape=(nb_hyperedges, d_edge_attr)
        :param node_features_per_node: tuple with *all* node features (individual_features, tissue_features, metagene_features):
                                       - individual_features: (nb_patients, d_patient)
                                       - tissue_features: (nb_tissues, d_tissue)
                                       - metagene_features: (nb_metagenes, d_metagene)
        :return: Node feature updates (tuple) and hyperedge attributes
        """
        dynamic_node_features, static_node_features = node_features_per_node

        # Expanded node features
        dynamic_node_features_e = {}
        static_node_features_e = {}
        for k, v in hyperedge_index.items():
            if k in dynamic_node_features:
                dynamic_node_features_e[k] = dynamic_node_features[k][v]
            elif k in static_node_features:
                static_node_features_e[k] = static_node_features[k][v]
        node_features_e = {**dynamic_node_features_e, **static_node_features_e}

        # Message passing
        m = self.messages(hyperedge_attr, node_features_e)
        m_updates = self.aggregate(m, hyperedge_index, dynamic_node_features_e, static_node_features_e, dynamic_node_features)
        dynamic_node_features = self.update(dynamic_node_features, m_updates)

        # Update hyperedge attributes
        if self.update_edge_attr:
            hyperedge_attr_updates = self.hyperedge_attr_update(hyperedge_attr, node_features_e)
            hyperedge_attr = hyperedge_attr + hyperedge_attr_updates

        # Return updates per node
        return dynamic_node_features, hyperedge_attr

    def messages(self, hyperedge_attr, node_features_e):
        """
        Computes individual messages
        :param hyperedge_attr: hyperedge features (similar to edge_attr of PyTorch Geometric). Shape=(nb_hyperedges, d_edge_attr)
        :param node_features_e: features of all hyperedges. Tuple (patient_features, tissue_features, gene_features) where:
                              - patient_features: patient features of each hyperedge. Shape=(nb_hyperedges, d_patient)
                              - tissue_features: tissue features of each hyperedge. Shape=(nb_hyperedges, d_tissue)
                              - gene_features: patient features of each hyperedge. Shape=(nb_hyperedges, d_metagene)
        :return: Torch tensor with messages to individuals. Shape=(nb_hyperedges, message_dim)
        """
        # Compute messages
        # print({k: node_features_e[k].shape for k in node_features_e.keys()})
        # print({k: node_features_e[k].get_device() for k in node_features_e.keys()})
        catted_features = torch.cat([node_features_e[k] for k in sorted(node_features_e.keys())]
                                    + [hyperedge_attr], dim=-1)
        messages = self.edge_mlp(catted_features)
        return messages

    def aggregate(self, messages, hyperedge_index, dynamic_node_features_e, static_node_features_e, dynamic_node_features):
        """
        Aggregates all the messages sent to the same individual
        :param messages: torch tensor with shape (nb_hyperedges, message_dim)
        :param hyperedge_index: indices of hyperedges (similar to edge_index of PyTorch Geometric). Shape=(3, nb_hyperedges)
        :param node_features: features of all hyperedges. Tuple (patient_features, tissue_features, gene_features) where:
                      - patient_features: patient features of each hyperedge. Shape=(nb_hyperedges, d_patient)
                      - tissue_features: tissue features of each hyperedge. Shape=(nb_hyperedges, d_tissue)
                      - gene_features: patient features of each hyperedge. Shape=(nb_hyperedges, d_metagene)
        :return: Aggregated messages. Shape=(nb_individuals, message_dim)
        """
        aggregated_messages = {}
        for k in dynamic_node_features_e.keys():
            
            # Aggregate messages
            m = message_aggregation(messages=messages,
                                    idxs=hyperedge_index[k],
                                    dim_size=dynamic_node_features[k].shape[0],  # Number of nodes of original tensor
                                    aggregators=self.aggregators)
            aggregated_messages[k] = m

        return aggregated_messages

    def update(self, dynamic_node_features, message_updates):
        """
        Computes node feature updates (currently individuals only).
        TODO: Update node features directly?
        :param node_features_per_node: tuple with *all* node features (individual_features, tissue_features, metagene_features):
                                       - individual_features: (nb_patients, d_patient)
                                       - tissue_features: (nb_tissues, d_tissue)
                                       - metagene_features: (nb_metagenes, d_metagene)
        :param message_updates: Aggregated messages. Shape=(nb_individuals, message_dim)
        :return: Node feature updates (currently individuals only). Shape=(nb_individuals, d_patient)
        """
        for k in dynamic_node_features.keys():
            h = torch.cat((dynamic_node_features[k], message_updates[k]), dim=-1)
            dynamic_node_features[k] = self.dynamic_node_mlp[k](h)

        return dynamic_node_features

    def hyperedge_attr_update(self, hyperedge_attr, node_features_e):
        """
        Updates hyperedge attributes
        :param hyperedge_attr: hyperedge features (similar to edge_attr of PyTorch Geometric). Shape=(nb_hyperedges, d_edge_attr)
        :param node_features: features of all hyperedges. Tuple (patient_features, tissue_features, gene_features) where:
                      - patient_features: patient features of each hyperedge. Shape=(nb_hyperedges, d_patient)
                      - tissue_features: tissue features of each hyperedge. Shape=(nb_hyperedges, d_tissue)
                      - gene_features: patient features of each hyperedge. Shape=(nb_hyperedges, d_metagene)
        :return: updated hyperedge attributes. Shape=(nb_hyperedges, d_edge_attr)
        """
        assert self.update_edge_attr

        # Compute messages
        catted_features = torch.cat([node_features_e[k] for k in sorted(node_features_e.keys())]
                                    + [hyperedge_attr], dim=-1)
        hyperedge_attr_updates = self.hyperedge_attr_mlp(catted_features)

        return hyperedge_attr_updates
