"""
Train/eval loops
"""

import torch
import wandb
import torch.nn.functional as F
from src.data_utils import sparsify, densify
from src.losses import get_reconstruction_loss
from torch.distributions import kl_divergence as kl
from torch.distributions import Normal


def train(config, model, loader, val_loader, epochs=None, use_wandb=True, **kwargs):
    """
    Trains the model
    :param config: Config object (e.g. Wandb config) with hyperparameters
    :param model: Model to train
    :param loader: Train loader
    :param val_loader: Validation loader
    :param use_wandb: whether to log the statistics into wandb
    :param kwargs: keyword arguments for the train and evaluate methods
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min',
                                                           factor=0.99, patience=5,
                                                           min_lr=0.00001)

    # Train/eval loop
    if epochs is None:
        epochs = config.epochs
    for epoch in range(epochs):
        losses = train_step(model=model,
                            optimiser=optimiser,
                            loader=loader,
                            beta=config.beta, **kwargs)
        scheduler.step(losses['loss'])
        losses_dict = losses

        if val_loader is not None:
            val_losses = eval_step(model=model,
                                   loader=val_loader,
                                   beta=config.beta, **kwargs)

            for k, v in val_losses.items():
                losses_dict[f'val_{k}'] = v

        if use_wandb:
            wandb.log(losses_dict)
        print(f'Epoch {epoch + 1}/{config.epochs}. ' + '. '.join([f'{k}: {v:.3f}' for k, v in losses_dict.items()]))


def train_step(model, optimiser, loader, **kwargs):
    """
    Performs one training step (i.e. one epoch)
    :param model: Model to train
    :param optimiser: Torch optimiser
    :param loader: Train loader
    :param kwargs: keyword arguments (currently unused)
    :return: epoch's train loss
    """
    model.train()
    losses_all = {}
    for data in loader:
        optimiser.zero_grad()
        out, node_features = forward(data, model, **kwargs)
        losses = compute_loss(data, out, node_features, **kwargs)

        for k, v in losses.items():
            if k in losses_all:
                losses_all[k] += v.item()
            else:
                losses_all[k] = v.item()

        # Backpropagate
        loss = losses['loss']
        loss.backward()
        optimiser.step()

    return {k: v / len(loader) for k, v in losses_all.items()}


def eval_step(model, loader, **kwargs):
    """
    Performs evaluation step
    :param model: Model to evaluate
    :param optimiser: Torch optimiser
    :param loader: Validation loader
    :param kwargs: keyword arguments (currently unused)
    :return: losses
    """
    model.eval()
    losses_all = {}
    with torch.no_grad():
        for data in loader:
            out, node_features = forward(data, model, **kwargs)
            losses = compute_loss(data, out, node_features, **kwargs)
            metrics = compute_metrics(data, out, node_features, **kwargs)
            losses = {**losses, **metrics}

            for k, v in losses.items():
                if k in losses_all:
                    losses_all[k] += v.item()
                else:
                    losses_all[k] = v.item()

    return {k: v / len(loader) for k, v in losses_all.items()}


def encode(data, model, preprocess_fn=None, **kwargs):
    """
    Produces features of nodes in the hypergraph
    :param data: Data object to be fed to the model
    :param model: Hypergraph model
    :param preprocess_fn: Function that processes the input data
    :return: Node features of nodes appearing in data
    """
    x_source = data.x_source
    if preprocess_fn is not None:
        # Compute log1p (just for input data)
        x_source = preprocess_fn(data.x_source)

    # Prediction model
    x_source = model.encode_metagenes(x_source)

    # Sparsify data
    metagenes = model.metagenes
    hyperedge_index, hyperedge_attr = sparsify(data.source, metagenes, x=x_source)

    # Compute node features
    node_features = model(hyperedge_index, hyperedge_attr, dynamic_node_features=data.node_features)

    return node_features


def decode(data, model, node_features, use_observed_library=True, n_cells=None, library=None, **kwargs):
    """
    Decodes the target data according to data.target.
    :param data: Data object (only information about the target nodes is used).
                 data.target is a dictionary mapping node types to lists of node indices for that type
                 (i.e. similar to edge index in pytorch geometric, but with named node types)
    :param model: Hypergraph model
    :param node_features: Encoded node features
    :param use_observed_library: Whether to use observed library sizes or predict them
    :return: Dictionary with parameters of generative model
    """
    metagenes = model.metagenes
    target_hyperedge_index, _ = sparsify(data.target, metagenes, x=None)

    # Compute predictions for each metagene in the target tissues
    x_pred_metagenes = model.predict(target_hyperedge_index, node_features,
                                     **kwargs)  # Out shape=(nb_metagenes, metagene_dim)

    # Densify data
    x_pred_metagenes = densify(data.target, metagenes, target_hyperedge_index, x_pred_metagenes)

    # Factor that multiplies library size (i.e. number of cells in the summed signature). For deconvolution experiment,
    # we set this value (extrinsic to the model) to the number of cells of the summed signature at train time. This is
    # because averaging signatures result in "non-integer counts" and so NB/ZINB losses cannot be used. At test time,
    # we predict the "average" signatures (the number of cells in the signature is 1, i.e. n_cells=1)
    use_observed_n_cells = n_cells == 0
    if use_observed_n_cells:
        n_cells = data.target_misc['n_cells'][:, None]

    log_library = None
    if use_observed_library:
        if library is None:
            library = data.x_target.sum(dim=-1, keepdims=True)
        if n_cells is not None:
            library = library / data.target_misc['n_cells'][:, None]
        log_library = torch.log(library)  # [:, None]

    # Map metagene features back to high-dimensional space
    out = model.decode_metagenes(x_pred_metagenes, log_library=log_library, n_cells=n_cells, **kwargs)

    return out


def forward(data, model, device=None, preprocess_fn=None,
            use_observed_library=True, use_latent_means=False, **kwargs):
    """
    Performs forward step on data
    :param data: Data to be fed to the model
    :param model: Pytorch model
    :param device: Pytorch device
    :param preprocess_fn: Function that processes the input data
    :param use_observed_library: Whether to use observed library sizes or predict them
    :param use_latent_means: Whether to use means of latent distribution (i.e. instead of sampling)
    :param kwargs: keyword arguments (currently unused)
    :return: predictions, loss, and node features (individual features, tissue features, metagene features)
    """
    # Map data to device
    data = data.to(device)

    # Compute node features
    node_features = encode(data, model, preprocess_fn=preprocess_fn)

    # Set latent variables to mean value (i.e. instead of sampling)
    if use_latent_means:
        (dynamic_node_features, static_node_features) = node_features
        for k in dynamic_node_features.keys():
            dynamic_node_features[k]['latent'] = dynamic_node_features[k]['mu']
        node_features = (dynamic_node_features, static_node_features)

    # Decode
    out = decode(data, model, node_features, use_observed_library=use_observed_library, **kwargs)

    return out, node_features


def compute_loss(data, out, node_features, beta=1., **kwargs):
    """
    Computes VAE loss
    :param data: Data object with ground truth targets
    :param out: Output dict from the model
    :param node_features: Encoded node features
    :param beta: beta hyperparameter of beta-VAE
    :return: dictionary of losses
    """

    rec_loss = torch.mean(get_reconstruction_loss(data.x_target, **out))

    # Compute loss
    loss = rec_loss
    out_dict = {'loss': loss}

    return out_dict


def compute_metrics(data, out, node_features, metric_fns=None, **kwargs):
    out_dict = {}

    if metric_fns is not None:
        for metric_fn in metric_fns:
            out_dict[metric_fn.__name__] = metric_fn(data.x_target.detach().cpu().numpy(), out)

    return out_dict
