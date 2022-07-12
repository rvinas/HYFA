"""
Decoders mapping metagenes to genes
"""
import torch
import torch.nn as nn
from src.hnn_utils import *
from torch.distributions import Normal


def get_decoder(name):
    if name == 'normal':
        return PlainDecoder
    elif name == 'poisson':
        return PoissonDecoder
    elif name == 'negative_binomial':
        return NegativeBinomialDecoder
    elif name == 'zero_inflated_negative_binomial':
        return ZeroInflatedNegativeBinomialDecoder
    else:
        raise ValueError(f'Decoder {name} not recognised')


class PlainDecoder(torch.nn.Module):
    """
    Linear mapping
    """

    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super(PlainDecoder, self).__init__()
        # self.px_decoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())

        self.px_rate_decoder = nn.Sequential(nn.Linear(in_dim, out_dim))

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(in_dim, out_dim)

    def forward(self, x, **kwargs):
        # px = self.px_decoder(x)
        px_rate = self.px_rate_decoder(x)
        px_r = nn.functional.softplus(self.px_r_decoder(x)) + 0.0001  # torch.exp(self.px_r_decoder(px))

        return {'px_r': px_r, 'px_rate': px_rate, 'gene_likelihood': 'normal'}


class PoissonDecoder(torch.nn.Module):
    """
    Linear mapping
    """

    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super(PoissonDecoder, self).__init__()
        self.px_decoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax(dim=-1),
        )

        # Compute library size
        self.library_mean = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.library_var = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.var_eps = 1e-4

    def forward(self, x, log_library=None, **kwargs):
        """
        Reference: SCVI-tools.
        See: https://github.com/scverse/scvi-tools/blob/2472b76572672fb9c98755522fa3c7326a946444/scvi/nn/_base_components.py#L374
        :param x:
        :param kwargs:
        :return:
        """
        px = self.px_decoder(x)

        # Compute library size if not given
        log_library_mean = None
        log_library_var = None
        if log_library is None:  # not using observed library
            log_library_mean = self.library_mean(px)
            log_library_var = torch.exp(self.library_var(px)) + self.var_eps
            log_library = Normal(log_library_mean, log_library_var.sqrt()).rsample()

        px_scale = self.px_scale_decoder(px)
        # px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(log_library) * px_scale  # torch.clamp( , max=12)
        return {'px_scale': px_scale, 'px_rate': px_rate,
                'library_mean': log_library_mean, 'library_var': log_library_var, 'library': log_library,
                'gene_likelihood': 'poisson'}


class NegativeBinomialDecoder(torch.nn.Module):
    """
    Linear mapping
    """

    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super(NegativeBinomialDecoder, self).__init__()
        self.px_decoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax(dim=-1),
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(hidden_dim, out_dim)

        # Compute library size
        self.library_mean = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.library_var = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.var_eps = 1e-4

    def forward(self, x, log_library=None, **kwargs):
        """
        Reference: SCVI-tools.
        See: https://github.com/scverse/scvi-tools/blob/2472b76572672fb9c98755522fa3c7326a946444/scvi/nn/_base_components.py#L374
        :param x:
        :param kwargs:
        :return:
        """
        px = self.px_decoder(x)

        # Compute library size if not given
        log_library_mean = None
        log_library_var = None
        if log_library is None:  # not using observed library
            log_library_mean = self.library_mean(px)
            log_library_var = torch.exp(self.library_var(px)) + self.var_eps
            log_library = Normal(log_library_mean, log_library_var.sqrt()).rsample()

        px_scale = self.px_scale_decoder(px)
        # px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(log_library) * px_scale  # torch.clamp( , max=12)
        px_r = torch.exp(torch.clamp(self.px_r_decoder(px), max=12))
        return {'px_scale': px_scale, 'px_r': px_r, 'px_rate': px_rate,
                'library_mean': log_library_mean, 'library_var': log_library_var, 'library': log_library,
                'gene_likelihood': 'nb'}


class ZeroInflatedNegativeBinomialDecoder(torch.nn.Module):
    """
    Linear mapping
    """

    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super(ZeroInflatedNegativeBinomialDecoder, self).__init__()
        self.px_decoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax(dim=-1),
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(hidden_dim, out_dim)

        # dropout
        self.px_dropout_decoder = nn.Linear(hidden_dim, out_dim)

        # Compute library size
        self.library_mean = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.library_var = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.var_eps = 1e-4

    def forward(self, x, log_library, n_cells=1., use_library_mean=False, **kwargs):
        """
        Reference: SCVI-tools.
        See: https://github.com/scverse/scvi-tools/blob/2472b76572672fb9c98755522fa3c7326a946444/scvi/nn/_base_components.py#L374
        :param x:
        :param kwargs:
        :return:
        """
        px = self.px_decoder(x)

        # Compute library size if not given
        log_library_mean = None
        log_library_var = None
        if log_library is None:  # not using observed library
            log_library_mean = self.library_mean(px)
            log_library_var = torch.exp(self.library_var(px)) + self.var_eps
            log_library = Normal(log_library_mean, log_library_var.sqrt()).rsample()

            if use_library_mean:
                log_library = log_library_mean

        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = n_cells * torch.exp(log_library) * px_scale  # torch.clamp( , max=12)
        px_r = torch.exp(torch.clamp(self.px_r_decoder(px), max=12))
        return {'px_scale': px_scale, 'px_r': px_r, 'px_rate': px_rate, 'px_dropout': px_dropout,
                'library_mean': log_library_mean, 'library_var': log_library_var, 'library': log_library,
                'gene_likelihood': 'zinb'}
