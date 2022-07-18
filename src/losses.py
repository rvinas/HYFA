"""
Adapted from SCVI-tools
Copyright (c) 2020 Romain Lopez, Adam Gayoso, Galen Xing, Yosef Lab
Copyright (c) 2022 Ramon Vinas
All rights reserved.
"""
from src.distributions import Poisson, NegativeBinomial, ZeroInflatedNegativeBinomial
import torch.nn.functional as F
import torch


def get_reconstruction_loss(x, px_rate, px_r=None, px_dropout=None, gene_likelihood='nb', aggr='mean',
                            **kwargs) -> torch.Tensor:
    if gene_likelihood == "zinb":
        reconst_loss = (
            -ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout
            )
                .log_prob(x)
        )
    elif gene_likelihood == "nb":
        reconst_loss = (
            -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x)
        )
    elif gene_likelihood == "poisson":
        reconst_loss = -Poisson(px_rate).log_prob(x)
    elif gene_likelihood == "normal":  # For normalised gene expression
        reconst_loss = -torch.distributions.normal.Normal(loc=px_rate, scale=px_r).log_prob(x)
        # reconst_loss = F.mse_loss(px_rate, x)  # Note: Ignoring sd

    if aggr == 'mean':
        reconst_loss = reconst_loss.mean(dim=-1)
    else:
        reconst_loss = reconst_loss.sum(dim=-1)

    return reconst_loss
