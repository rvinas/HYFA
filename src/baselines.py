from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import KNNImputer, SimpleImputer
import numpy as np


def PCA_linear_regression_baseline(x_source, x_target, x_source_test, x_source_covs=None, x_source_test_covs=None,
                                   n_components=30, verbose=False):
    """
    1. PCA on source tissue
    2. Linear regression from source components to target gene expression
    """
    # Centering
    means = x_source.mean(axis=0, keepdims=True)
    x_source_ = (x_source.copy() - means)
    x_source_test_ = (x_source_test.copy() - means)

    # SVD for source gene expression
    svd_source = TruncatedSVD(n_components=n_components)
    x_source_lowd = svd_source.fit_transform(x_source_)
    x_source_lowd_test = svd_source.transform(x_source_test_)
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