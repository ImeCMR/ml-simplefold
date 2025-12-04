#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
from tqdm import tqdm
from einops import repeat
from utils.boltz_utils import center_random_augmentation
from .bayesian_steering import BayesianSteering


class EMSampler():
    """
    A Euler-Maruyama solver for SDEs.
    """
    def __init__(
        self,
        num_timesteps=500,
        t_start=1e-4,
        tau=0.3,
        log_timesteps=False,
        w_cutoff=0.99,
        use_bayesian_steering=False,
        bayesian_steering_params=None,
        gamma_t_schedule=[0.1, 1.0],
    ):
        self.num_timesteps = num_timesteps
        self.log_timesteps = log_timesteps
        self.t_start = t_start
        self.tau = tau
        self.w_cutoff = w_cutoff
        self.use_bayesian_steering = use_bayesian_steering
        self.gamma_t_schedule = gamma_t_schedule

        if self.use_bayesian_steering:
            self.bayesian_steering = bayesian_steering_params

        if self.log_timesteps:
            t = 1.0 - torch.logspace(-2, 0, self.num_timesteps + 1).flip(0)
            t = t - torch.min(t)
            t = t / torch.max(t)
            self.steps = t.clamp(min=self.t_start, max=1.0)
        else:
            self.steps = torch.linspace(
                self.t_start, 1.0, steps=self.num_timesteps + 1
            )

    def diffusion_coefficient(self, t, eps=0.01):
        # determine diffusion coefficient
        w = (1.0 - t) / (t + eps)
        if t >= self.w_cutoff:
            w = 0.0
        return w

    def get_gamma_t(self, t):
        # Time-dependent steering schedule (gamma(t))
        start, end = self.gamma_t_schedule
        return start + (end - start) * t

    def euler_maruyama_step(
        self,
        model_fn,
        flow,
        y, 
        t, 
        t_next, 
        batch, 
    ):
        dt = t_next - t
        eps = torch.randn_like(y).to(y)

        y = center_random_augmentation(
            y,
            batch["atom_pad_mask"],
            augmentation=False,
            centering=True,
        )

        with torch.no_grad():
            batched_t = repeat(t, " -> b", b=y.shape[0])
            velocity = model_fn(
                noised_pos=y,
                t=batched_t,
                feats=batch,
            )['predict_velocity']
            score = flow.compute_score_from_velocity(velocity, y, t)
            diff_coeff = self.diffusion_coefficient(t)
            drift = velocity + diff_coeff * score

        if self.use_bayesian_steering:
            with torch.enable_grad():
                y_steer = y.detach().clone().requires_grad_(True)
                E_bayesian = self.bayesian_steering(y_steer, batch, t)
                F_steering = -torch.autograd.grad(E_bayesian, y_steer)[0]

            gamma_t = self.get_gamma_t(t)
            drift = drift + gamma_t * F_steering

        with torch.no_grad():
            mean_y = y + drift * dt
            y_sample = mean_y + torch.sqrt(2.0 * dt * diff_coeff * self.tau) * eps

        return y_sample

    @torch.no_grad()
    def sample(self, model_fn, flow, noise, batch):
        sampling_timesteps = self.num_timesteps
        steps = self.steps.to(noise.device)
        y_sampled = noise
        feats = batch

        for i in tqdm(
            range(sampling_timesteps),
            desc="Sampling",
            total=sampling_timesteps,
        ):
            t = steps[i]
            t_next = steps[i + 1]

            y_sampled = self.euler_maruyama_step(
                model_fn,
                flow,
                y_sampled,
                t,
                t_next,
                feats,
            )

        return {
            "denoised_coords": y_sampled
        }
