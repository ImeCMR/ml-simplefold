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
        log_timestep=False,
        w_cutoff=0.99,
        bayesian_steering: BayesianSteering = None,
        steering_schedule_gamma: float = 1.0,
        steering_schedule_type: str = 'linear',
    ):
        self.num_timesteps = num_timesteps
        self.log_timesteps = log_timestep
        self.t_start = t_start
        self.tau = tau
        self.w_cutoff = w_cutoff
        self.bayesian_steering = bayesian_steering
        self.steering_schedule_gamma = steering_schedule_gamma
        self.steering_schedule_type = steering_schedule_type

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

    @torch.no_grad()
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

        batched_t = repeat(t, " -> b", b=y.shape[0])
        velocity = model_fn(
            noised_pos=y,
            t=batched_t,
            feats=batch,
        )['predict_velocity']
        score = flow.compute_score_from_velocity(velocity, y, t)

        f_steering = torch.zeros_like(y)
        if self.bayesian_steering is not None and 'noesy_restraints' in batch and batch['noesy_restraints']:
            # Use per-item metadata from batch
            atom_to_idx_list = batch.get('atom_to_idx_list', [])
            bonds_list = batch.get('bonds_list', [])
            atom_names_list = batch.get('atom_names_list', [])

            f_steering_list = []
            for batch_idx in range(y.shape[0]):
                restraints_item = batch['noesy_restraints'][batch_idx]

                if not restraints_item or not atom_to_idx_list:
                    f_steering_list.append(torch.zeros_like(y[batch_idx]))
                    continue

                # Get per item data
                atom_to_idx_item = atom_to_idx_list[batch_idx]
                bonds_item = bonds_list[batch_idx] if bonds_list else []
                atom_names_item = atom_names_list[batch_idx] if atom_names_list else []

                # Compute steering force for this item
                f_steering_item, _ = self.bayesian_steering(
                    y[batch_idx:batch_idx+1],
                    [restraints_item], # Wrap in list
                    t,
                    bonds_item,
                    atom_to_idx_item,
                    atom_names_item
                )

                f_steering_list.append(f_steering_item.squeeze(0))

            if f_steering_list:
                f_steering = torch.stack(f_steering_list, dim=0)

        gamma_t = self.steering_schedule(t)

        diff_coeff = self.diffusion_coefficient(t)
        drift = velocity + diff_coeff * score + gamma_t * f_steering
        mean_y = y + drift * dt
        y_sample = mean_y + torch.sqrt(2.0 * dt * diff_coeff * self.tau) * eps

        return y_sample

    def steering_schedule(self, t):
        # Implement multiple schedule types
        if self.steering_schedule_type == 'linear':
            # Linear ramp: starts at 0, increases to gamma at t=1
            return t * self.steering_schedule_gamma
        
        elif self.steering_schedule_type == 'sigmoid':
            # Sigmoid: smooth ramp centered at t=0.5
            return torch.sigmoid(5 * (t - 0.5)) * self.steering_schedule_gamma
        
        elif self.steering_schedule_type == 'exponential':
            # Exponential: slowly increases early, faster later
            return (torch.exp(t) - 1) / (torch.e - 1) * self.steering_schedule_gamma
        
        elif self.steering_schedule_type == 'constant':
            # Constatnt: always gamma
            return torch.ones_like(t) * self.steering_schedule_gamma
        
        elif self.steering_schedule_type == 'square':
            # Quadratic: t^2 schedule
            return (t ** 2) * self.steering_schedule_gamma
        
        else:
            raise ValueError(f"Unknown steering schedule type: {self.steering_schedule_type}")

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
