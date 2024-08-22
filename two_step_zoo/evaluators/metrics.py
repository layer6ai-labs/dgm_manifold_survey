import torch
import numpy as np
from pytorch_fid import fid_score
import prdc

from .metrics_helpers import FDHelper
from .ood_helpers import ood_acc


def fd(module, eval_loader=None, train_loader=None, gen_samples=50000, gen_batch_size=64,
       cache=None, use_dinov2=True, **kwargs):
    """
    Following Heusel et al. (2017), compute FD from the training set if provided, except defaults to DINOv2 instead of
    the Inception network
    """
    dataloader = eval_loader if train_loader is None else train_loader
    fd_helper = FDHelper(module, dataloader, gen_samples, gen_batch_size, use_dinov2)

    gen_mu, gen_sigma = fd_helper.compute_fd_stats(**kwargs)

    if cache is None:
        gt_mu, gt_sigma = fd_helper.compute_fd_stats(dataloader)
    elif "gt_feats" not in cache:
        gt_feats = fd_helper.get_fd_features(dataloader)
        cache["gt_feats"] = gt_feats
        gt_mu = np.mean(gt_feats, axis=0)
        gt_sigma = np.cov(gt_feats, rowvar=False)
        cache["gt_stats"] = gt_mu, gt_sigma
    elif "gt_stats" not in cache:
        gt_feats = cache["gt_feats"]
        gt_mu = np.mean(gt_feats, axis=0)
        gt_sigma = np.cov(gt_feats, rowvar=False)
        cache["gt_stats"] = gt_mu, gt_sigma
    else:
        gt_mu, gt_sigma = cache["gt_stats"]

    return fid_score.calculate_frechet_distance(gen_mu, gen_sigma, gt_mu, gt_sigma)


def precision_recall_density_coverage(module, eval_loader=None, train_loader=None, gen_samples=50000, gen_batch_size=64,
                                      nearest_k=5, cache=None, use_dinov2=True, **kwargs):
    """
    Following Naaem et al. (2020), compute Precision, Recall, Density, Coverage from the training set if provided.
    """
    dataloader = eval_loader if train_loader is None else train_loader
    fd_helper = FDHelper(module, dataloader, gen_samples, gen_batch_size, use_dinov2)

    gen_feats = fd_helper.get_fd_features(**kwargs)

    if cache is None:
        gt_feats = fd_helper.get_fd_features(dataloader)
    elif "gt_feats" not in cache:
        gt_feats = fd_helper.get_fd_features(dataloader)
        cache["gt_feats"] = gt_feats
    else:
        gt_feats = cache["gt_feats"]

    return prdc.compute_prdc(gt_feats, gen_feats, nearest_k)


def log_likelihood(module, dataloader, cache=None):
    with torch.no_grad():
        return module.log_prob(dataloader).mean()


def l2_reconstruction_error(module, dataloader, cache=None):
    with torch.no_grad():
        return module.rec_error(dataloader).mean()


def loss(module, dataloader, cache=None):
    with torch.no_grad():
        return module.loss(dataloader).mean()


def null_metric(module, dataloader, cache=None):
    return 0


def likelihood_ood_acc(
        module,
        is_test_loader,
        oos_test_loader,
        is_train_loader,
        oos_train_loader,
        savedir,
        cache=None,
    ):
    return ood_acc(
        module, is_test_loader, oos_test_loader, is_train_loader, oos_train_loader, savedir,
        low_dim=False, cache=cache
    )


def likelihood_ood_acc_low_dim(
        module,
        is_test_loader,
        oos_test_loader,
        is_train_loader,
        oos_train_loader,
        savedir,
        cache=None,
    ):
    return ood_acc(
        module, is_test_loader, oos_test_loader, is_train_loader, oos_train_loader, savedir,
        low_dim=True, cache=cache
    )
