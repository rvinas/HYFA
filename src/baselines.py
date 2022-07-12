from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import KNNImputer, SimpleImputer
from src.data import Data
from src.dataset import HypergraphDataset
from torch.utils.data import DataLoader
from src.hnn import HypergraphNeuralNet
from collections import Counter
from src.train_utils import train, forward
import numpy as np
import scanpy as sc
from tqdm import tqdm
import torch


def PCA_linear_regression_baseline_v1(x_source, x_target, x_source_test, x_source_covs=None, x_source_test_covs=None,
                                      n_components=30, verbose=False):
    """
    1. PCA on source tissue
    2. Linear regression from source components to target gene expression
    """
    # SVD for source gene expression
    svd_source = TruncatedSVD(n_components=n_components)
    x_source_lowd = svd_source.fit_transform(x_source)
    x_source_lowd_test = svd_source.transform(x_source_test)
    if verbose:
        print('SVD explained variance (source): ', svd_source.explained_variance_ratio_.sum())

    # Append covariates
    if x_source_covs is not None:
        assert x_source_test_covs is not None
        x_source_lowd = np.concatenate((x_source_lowd, x_source_covs), axis=-1)
        x_source_lowd_test = np.concatenate((x_source_lowd_test, x_source_test_covs), axis=-1)

    # Linear regression
    reg = LinearRegression().fit(x_source_lowd, x_target)

    # Make predictions
    x_target_pred = reg.predict(x_source_lowd_test)

    return x_target_pred


# =================================
# Baselines
# =================================

def impute_knn(y_observed, covariates, k=10):
    imputer = KNNImputer(n_neighbors=k)
    y_observed_ = y_observed.reshape((y_observed.shape[0], -1))
    y_observed_ = np.concatenate((y_observed_, covariates), axis=-1)
    y_imp_ = imputer.fit_transform(y_observed_)
    y_imp = y_imp_[:, :-covariates.shape[-1]].reshape(y_observed.shape)
    return y_imp


def impute_simple(y_observed, covariates, strategy='mean'):
    N, T, G = y_observed.shape

    # Perform initial imputation on unobserved tissues
    y_imp = SimpleImputer(missing_values=np.nan,
                          strategy=strategy).fit_transform(y_observed.reshape(N, T * G)).reshape(N, T, G)
    return y_imp


def impute_TEEBoT(y_observed, covariates, n_components=10, strategy='mean'):
    N, T, G = y_observed.shape
    mask = np.isnan(y_observed)[..., 0]  # Shape=(N, T)

    # Perform initial imputation on unobserved tissues
    y_observed_ = SimpleImputer(missing_values=np.nan,
                                strategy=strategy).fit_transform(y_observed.reshape(N, T * G)).reshape(N, T, G)
    y_imp = np.copy(y_observed)
    for t in tqdm(range(T)):
        not_t = [i for i in range(T) if i != t]

        # Gather observed and missing samples from tissue t
        mask_t = mask[:, t]  # Shape=(N,)
        y_t_target = y_observed_[mask_t, :][:, t]  # Shape=(N_observed, G)
        y_t_source = y_observed_[mask_t, :][:, not_t].reshape(y_t_target.shape[0], -1)

        y_t_pred = PCA_linear_regression_baseline_v1(y_t_source,
                                                     y_t_target,
                                                     y_t_source,
                                                     x_source_covs=covariates[mask_t],
                                                     x_source_test_covs=covariates[mask_t],
                                                     n_components=n_components,
                                                     verbose=False)
        y_imp[mask_t, t] = y_t_pred

    return y_imp