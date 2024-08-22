import numpy as np
import torch

from .density_estimator import DensityEstimator
from .generalized_autoencoder import GeneralizedAutoEncoder
from .utils import batch_or_dataloader
from .distributions import diagonal_gaussian_log_prob, diagonal_gaussian_entropy, diagonal_gaussian_sample


class GaussianVAE(GeneralizedAutoEncoder, DensityEstimator):
    model_type = "vae"

    def __init__(self, fixed_sigma=None, **kwargs):  # TODO: only handling fixed_sigma for VAEs, AVB has to be added
        super().__init__(**kwargs)
        self.fixed_sigma = fixed_sigma

    def sample(self, n_samples, true_sample=True):
        z = torch.randn(((n_samples,) + self.latent_shape)).to(self.device)
        if self.fixed_sigma is None:
            mu, log_sigma = self.decode_to_transformed(z)
        else:
            mu = self.decode_to_transformed(z)
            log_sigma = torch.ones_like(mu) * np.log(self.fixed_sigma)
        sample = diagonal_gaussian_sample(mu, torch.exp(log_sigma)) if true_sample else mu
        return self._inverse_data_transform(sample)

    @batch_or_dataloader()
    def log_prob(self, x, k=1):
        # NOTE: With k=1, this gives the ELBO.
        batch_size = x.shape[0]

        # NOTE: Perform data transform _before_ repeat_interleave because we do not want
        #       to dequantize the same x point in several different ways.
        x = self._data_transform(x)
        x = x.repeat_interleave(k, dim=0)

        mu_z, log_sigma_z = self.encode_transformed(x)

        z = diagonal_gaussian_sample(mu_z, torch.exp(log_sigma_z))
        if self.fixed_sigma is None:  # sigma_x is learnable and part of the output of the decoder
            mu_x, log_sigma_x = self.decode_to_transformed(z)
        else:  # sigma_x is provided as a hyperparameter, this is equivalent to a beta VAE
            mu_x = self.decode_to_transformed(z)
            log_sigma_x = torch.ones_like(mu_x) * np.log(self.fixed_sigma)

        log_p_z = diagonal_gaussian_log_prob(
            z.flatten(start_dim=1),
            torch.zeros_like(z).flatten(start_dim=1),
            torch.zeros_like(z).flatten(start_dim=1)
        )
        log_p_x_given_z = diagonal_gaussian_log_prob(
            x.flatten(start_dim=1),
            mu_x.flatten(start_dim=1),
            log_sigma_x.flatten(start_dim=1)
        )
        if k == 1:
            h_z_given_x = diagonal_gaussian_entropy(log_sigma_z.flatten(start_dim=1))
            return log_p_x_given_z + log_p_z + h_z_given_x
        else:
            log_q_z_given_x = diagonal_gaussian_log_prob(
                z.flatten(start_dim=1),
                mu_z.flatten(start_dim=1),
                log_sigma_z.flatten(start_dim=1)
            )
            elbo = log_p_z + log_p_x_given_z - log_q_z_given_x
            return torch.logsumexp(elbo.reshape(batch_size, k, 1), dim=1) - np.log(k)
