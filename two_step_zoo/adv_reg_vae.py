from itertools import chain
from torch.nn.functional import binary_cross_entropy_with_logits
from functorch import jvp, vjp

import numpy as np
import torch

from .density_estimator import DensityEstimator
from .generalized_autoencoder import GeneralizedAutoEncoder
from .utils import batch_or_dataloader
from .distributions import diagonal_gaussian_log_prob, diagonal_gaussian_entropy, diagonal_gaussian_sample


class AdversariallyRegularizedVAE(GeneralizedAutoEncoder, DensityEstimator):
    model_type = "adv_vae"

    def __init__(self, discriminator, beta1, beta2, num_discriminator_steps=2, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.beta1 = beta1
        self.beta2 = beta2
        self.num_discriminator_steps = num_discriminator_steps

        self.fixed_sigma = np.sqrt(2.) / 2.  # ensures Gaussian log likelihood is exactly l2 error
        self.step_count = 0
        self.last_ge_loss = torch.tensor(0.0)
        self.last_rec_error = torch.tensor(0.0).to(self.device)

    def train_batch(self, x, **kwargs):
        self.optimizer[0].zero_grad()
        discriminator_loss = self._discr_error_batch(x).mean()
        discriminator_loss.backward()
        self.optimizer[0].step()

        self.last_ge_loss = self.last_ge_loss.to(self.device)
        self.step_count += 1
        # NOTE: Take several steps for discriminator for each generator/encoder step
        if self.step_count >= self.num_discriminator_steps:
            self.step_count = 0

            self.optimizer[1].zero_grad()
            generator_encoder_loss = self._ge_error_batch(x).mean()
            generator_encoder_loss.backward()
            self.last_ge_loss = generator_encoder_loss
            self.optimizer[1].step()
            self.lr_scheduler[0].step()  # update schedulers together to prevent ge having larger lr after many epochs
            self.lr_scheduler[1].step()

        return {
            "discriminator_loss": discriminator_loss,
            "generator_encoder_loss": self.last_ge_loss,
        }

    def _stochastic_rec(self, x):
        mu_z, log_sigma_z = self.encode_transformed(x)
        z = diagonal_gaussian_sample(mu_z, torch.exp(log_sigma_z))
        mu_x = self.decode_to_transformed(z)
        return mu_z, log_sigma_z, z, mu_x

    @batch_or_dataloader()
    def log_prob(self, x, return_rec=False):
        # NOTE: this is not an actual log prob as for other density models, but the method is still called log_prob
        #       for consistency and compatibility
        x = self._data_transform(x)

        mu_z, log_sigma_z, z, mu_x = self._stochastic_rec(x)
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
        self.last_rec_error = log_p_x_given_z.mean()
        h_z_given_x = diagonal_gaussian_entropy(log_sigma_z.flatten(start_dim=1))
        out = log_p_x_given_z + self.beta1 * (log_p_z + h_z_given_x)
        if return_rec:
            return out, mu_x
        else:
            return out

    def _discr_error_batch_with_mu_x(self, x, mu_x):
        d_x = self.discriminator(x)
        d_rec = self.discriminator(mu_x)

        zeros = torch.zeros_like(d_rec)
        ones = torch.ones_like(d_x)

        # NOTE: Train discriminator to be positive on real data
        d_rec_correct = binary_cross_entropy_with_logits(d_rec, zeros)
        d_x_correct = binary_cross_entropy_with_logits(d_x, ones)

        out = d_rec_correct + d_x_correct
        with torch.no_grad():
            self.beta2 = torch.abs(self.last_rec_error) / (torch.abs(out.mean()) + 1e-6)

        return self.beta2 * out

    def _discr_error_batch(self, x):
        _, _, _, mu_x = self._stochastic_rec(x)
        return self._discr_error_batch_with_mu_x(x, mu_x)

    def _ge_error_batch(self, x):
        # VAE loss
        elbo, mu_x = self.log_prob(x, return_rec=True)

        # Discriminator loss
        discriminator_loss = self._discr_error_batch_with_mu_x(x, mu_x)

        return - (elbo + discriminator_loss)

    def sample(self, n_samples):
        z = torch.randn(((n_samples,) + self.latent_shape)).to(self.device)
        mu = self.decode_to_transformed(z)
        return self._inverse_data_transform(mu)

    def set_optimizer(self, cfg):
        disc_optimizer = self._OPTIMIZER_MAP[cfg["optimizer"]](
            self.discriminator.parameters(), lr=cfg["disc_lr"]
        )
        ge_optimizer = self._OPTIMIZER_MAP[cfg["optimizer"]](
            chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=cfg["ge_lr"]
        )
        self.optimizer = [disc_optimizer, ge_optimizer]
        self.num_optimizers = 2

        disc_lr_scheduler = self._get_lr_scheduler(
            optim=disc_optimizer,
            use_scheduler=cfg.get("use_disc_lr_scheduler", False),
            cfg=cfg
        )
        ge_lr_scheduler = self._get_lr_scheduler(
            optim=ge_optimizer,
            use_scheduler=cfg.get("use_ge_lr_scheduler", False),
            cfg=cfg
        )
        self.lr_scheduler = [disc_lr_scheduler, ge_lr_scheduler]
