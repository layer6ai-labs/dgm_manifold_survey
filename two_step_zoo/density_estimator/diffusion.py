import torch

from .density_estimator import DensityEstimator
from ..utils import batch_or_dataloader
from ..distributions import diagonal_gaussian_sample


class ScoreBasedDiffusionModel(DensityEstimator):

    model_type = "sbdm"

    def __init__(
            self,
            score_network,
            x_shape,
            T=1.,
            beta_min=0.1,
            beta_max=20,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.score_network = score_network
        self.x_shape = x_shape
        self.T = T
        self.beta_min = beta_min
        self.beta_diff = (beta_max - beta_min) / T

        # TODO: implement log_prob

    def sample(self, n_samples, eps=1e-2, steps=1000, track_score_norm=False):
        # Samples from the diffusion process using an Euler-Maruyama scheme, stops at time T-eps.
        with torch.no_grad():
            sample = torch.randn((n_samples,) + tuple(self.x_shape)).to(self.device)
            ts = torch.linspace(start=0., end=self.T - eps, steps=steps).to(self.device)
            delta_t = (self.T - eps) / (steps - 1)
            if track_score_norm:
                normalized_score_norms = []
            for t in ts:
                _, sigma_T_minus_t = self._get_sigma(self.T - t)
                score = self.score_network(sample, self.T - t.repeat(sample.shape[0])) / sigma_T_minus_t
                if track_score_norm:
                    norms = torch.mean(torch.square(score.flatten(start_dim=1)), dim=1).cpu().detach().numpy()
                    normalized_score_norms.append(norms)
                beta = self.beta_min + self.beta_diff * (self.T - t)
                z = torch.randn((n_samples,) + tuple(self.x_shape)).to(self.device)
                sample += delta_t * beta * (sample / 2. + score) + torch.sqrt(beta * delta_t) * z
            if track_score_norm:
                return self._inverse_data_transform(sample), normalized_score_norms
            else:
                return self._inverse_data_transform(sample)

    @batch_or_dataloader(agg_func=lambda x: torch.mean(torch.Tensor(x)))
    def loss(self, x):
        # Uses w(t)=sigma_t^2 as the weighting function.
        x = self._data_transform(x)
        t = self.T * torch.rand(x.shape[0]).to(self.device)
        eps = diagonal_gaussian_sample(torch.zeros_like(x), torch.ones_like(x))
        sigma2_t, sigma_t = self._get_sigma(t)
        x_input = torch.sqrt(1. - sigma2_t) * x + sigma_t * eps
        unnormalized_score = self.score_network(x_input, t)
        sq_error = torch.square(unnormalized_score + eps)
        loss = torch.sum(sq_error.flatten(start_dim=1), dim=1)
        return loss.mean()

    def _get_sigma(self, t):
        sigma2_t = 1.0 - torch.exp(- self.beta_min * t - self.beta_diff * t**2 / 2.)
        sigma2_t = torch.reshape(sigma2_t, (-1, 1, 1, 1))  # TODO: currently this only works for 4-D inputs
        return sigma2_t, torch.sqrt(sigma2_t)
