#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
Bayesian Steering Module for SDE Sampler

This module implements the modified Euler-Maruyama solver that incorporates
external steering forces from Bayesian energy functions during the SDE integration.
"""

import torch
from tqdm import tqdm
from typing import Callable, Optional, Dict, Any
from einops import repeat


class SteeringSchedule:
    """Time-dependent steering schedule γ(t)."""
    
    def __init__(
        self,
        schedule_type: str = "linear",
        start_t: float = 0.5,
        max_weight: float = 1.0,
    ):
        """
        Parameters
        ----------
        schedule_type : str
            Type of schedule: "linear", "sigmoid", or "constant"
        start_t : float
            Time at which to start steering (0 to 1, where 1 is clean data)
        max_weight : float
            Maximum weight at t=1
        """
        self.schedule_type = schedule_type
        self.start_t = start_t
        self.max_weight = max_weight
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute steering weight γ(t).
        
        Parameters
        ----------
        t : torch.Tensor
            Time step [B]
        
        Returns
        -------
        torch.Tensor
            Steering weight [B]
        """
        if self.schedule_type == "linear":
            # Linear ramp: γ(t) = 0 for t < start_t, linearly increase to max_weight at t=1
            weight = torch.zeros_like(t)
            mask = t >= self.start_t
            weight[mask] = self.max_weight * (t[mask] - self.start_t) / (1.0 - self.start_t)
            return weight
        
        elif self.schedule_type == "sigmoid":
            # Sigmoid schedule for smooth transition
            # γ(t) = max_weight / (1 + exp(-k(t - start_t)))
            k = 20.0  # Steepness
            weight = self.max_weight / (1.0 + torch.exp(-k * (t - self.start_t)))
            return weight
        
        elif self.schedule_type == "constant":
            # Constant weight if t >= start_t
            weight = torch.where(t >= self.start_t, self.max_weight * torch.ones_like(t), torch.zeros_like(t))
            return weight
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


class BayesianSteeredEMSampler:
    """
    Euler-Maruyama solver with Bayesian steering forces.
    
    This sampler modifies the standard SDE:
        dx_t = v_θ(x_t, s, t) dt + √(2τω(t)) dW_t
    
    To:
        dx_t = [v_θ(x_t, s, t) + γ(t)F_steering(x_t)] dt + √(2τω(t)) dW_t
    
    where F_steering = -∇_{x_t} E_Bayesian'(x_t) is computed externally.
    """
    
    def __init__(
        self,
        num_timesteps: int = 500,
        t_start: float = 1e-4,
        tau: float = 0.3,
        log_timesteps: bool = False,
        w_cutoff: float = 0.99,
        steering_schedule_type: str = "linear",
        steering_start_t: float = 0.5,
        steering_max_weight: float = 1.0,
    ):
        """
        Parameters
        ----------
        num_timesteps : int
            Number of integration steps
        t_start : float
            Minimum time step
        tau : float
            Diffusion coefficient
        log_timesteps : bool
            Use log spacing for timesteps
        w_cutoff : float
            Cutoff for diffusion coefficient
        steering_schedule_type : str
            Type of steering schedule
        steering_start_t : float
            When to start applying steering
        steering_max_weight : float
            Maximum steering weight
        """
        self.num_timesteps = num_timesteps
        self.t_start = t_start
        self.tau = tau
        self.w_cutoff = w_cutoff
        self.log_timesteps = log_timesteps
        
        if self.log_timesteps:
            t = 1.0 - torch.logspace(-2, 0, self.num_timesteps + 1).flip(0)
            t = t - torch.min(t)
            t = t / torch.max(t)
            self.steps = t.clamp(min=self.t_start, max=1.0)
        else:
            self.steps = torch.linspace(
                self.t_start, 1.0, steps=self.num_timesteps + 1
            )
        
        self.steering_schedule = SteeringSchedule(
            schedule_type=steering_schedule_type,
            start_t=steering_start_t,
            max_weight=steering_max_weight,
        )
    
    def diffusion_coefficient(self, t: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
        """Compute diffusion coefficient ω(t)."""
        w = (1.0 - t) / (t + eps)
        w = torch.where(t >= self.w_cutoff, torch.zeros_like(w), w)
        return w
    
    @torch.no_grad()
    def euler_maruyama_step(
        self,
        model_fn: Callable,
        flow: Any,
        y: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        batch: Dict,
        steering_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Single Euler-Maruyama integration step with optional steering.
        
        Parameters
        ----------
        model_fn : Callable
            Learned velocity field v_θ(y_t, s, t)
        flow : Any
            Flow path object (LinearPath)
        y : torch.Tensor
            Current coordinates [B, N_atoms, 3]
        t : torch.Tensor
            Current time [B]
        t_next : torch.Tensor
            Next time step
        batch : Dict
            Batch data
        steering_fn : Optional[Callable]
            Function that computes steering force gradient
            Should return [B, N_atoms, 3]
        
        Returns
        -------
        torch.Tensor
            Updated coordinates
        """
        from utils.boltz_utils import center_random_augmentation
        
        dt = t_next - t
        eps = torch.randn_like(y).to(y)
        
        y = center_random_augmentation(
            y,
            batch["atom_pad_mask"],
            augmentation=False,
            centering=True,
        )
        
        batched_t = repeat(t, " -> b", b=y.shape[0])
        
        # Compute learned velocity field v_θ
        velocity = model_fn(
            noised_pos=y,
            t=batched_t,
            feats=batch,
        )['predict_velocity']
        
        score = flow.compute_score_from_velocity(velocity, y, t)
        
        # Compute drift from learned velocity
        diff_coeff = self.diffusion_coefficient(t)
        drift = velocity + diff_coeff * score
        
        # Add steering force if provided
        if steering_fn is not None:
            steering_force = steering_fn(y, t)  # [B, N_atoms, 3]
            gamma = self.steering_schedule(t)  # [B]
            gamma = gamma.view(-1, 1, 1)  # [B, 1, 1]
            
            # Add scaled steering force to drift
            drift = drift + gamma * steering_force
        
        mean_y = y + drift * dt
        y_sample = mean_y + torch.sqrt(2.0 * dt * diff_coeff * self.tau) * eps
        
        return y_sample
    
    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable,
        flow: Any,
        noise: torch.Tensor,
        batch: Dict,
        steering_fn: Optional[Callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate samples using Euler-Maruyama integration with steering.
        
        Parameters
        ----------
        model_fn : Callable
            Learned velocity field
        flow : Any
            Flow path object
        noise : torch.Tensor
            Initial noise [B, N_atoms, 3]
        batch : Dict
            Batch data
        steering_fn : Optional[Callable]
            Steering force function
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with sampled coordinates
        """
        sampling_timesteps = self.num_timesteps
        steps = self.steps.to(noise.device)
        y_sampled = noise
        
        for i in tqdm(
            range(sampling_timesteps),
            desc="Sampling (Bayesian Steered)",
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
                batch,
                steering_fn=steering_fn,
            )
        
        return {
            "denoised_coords": y_sampled
        }
