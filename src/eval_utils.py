import numpy as np


# Metrics
def pearson_correlation(x, y):
    """
    Computes similarity measure between each pair of genes in the bipartite graph x <-> y
    :param x: Gene matrix 1. Shape=(nb_samples, nb_genes_1)
    :param y: Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    :return: Matrix with shape (nb_genes_1, nb_genes_2) containing the similarity coefficients
    """

    def standardize(a):
        a_off = np.mean(a, axis=0)
        a_std = np.std(a, axis=0)
        return (a - a_off) / a_std

    assert x.shape[0] == y.shape[0]
    x_ = standardize(x)
    y_ = standardize(y)
    return np.dot(x_.T, y_) / x.shape[0]


def r2(x_gt, x_pred):
    means = np.mean(x_gt, axis=0)  # Shape=(nb_genes,)
    ss_res = np.sum((x_gt - x_pred) ** 2, axis=0)
    ss_tot = np.sum((x_gt - means) ** 2, axis=0)
    r_sq = 1 - ss_res / ss_tot
    return r_sq


# Score functions
def pearson_correlation_score(x_gt, x_pred, sample_corr=False):
    if sample_corr:
        corrs = pearson_correlation(x_gt.T, x_pred.T)
    else:
        corrs = pearson_correlation(x_gt, x_pred)
    corr = np.diagonal(corrs)
    return corr


def r2_score(x_gt, x_pred, sample_corr=False):
    if sample_corr:
        score = r2(x_gt.T, x_pred.T)
    else:
        score = r2(x_gt, x_pred)
    return score


def compute_scores(examples, x_pred, score_fn):
    x_source, x_target, tissues_source, tissues_target, patients_source, patients_target = examples
    scores = score_fn(x_target.numpy(), x_pred)
    return scores
