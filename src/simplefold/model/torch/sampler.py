"""
Modified SimpleFold sampler with NMR guidance.
Extends the EMSampler with inference-time NMR restraint guidance.
"""
import torch
import math
from tqdm import tqdm
from einops import repeat
from typing import Dict, Optional
from utils.boltz_utils import center_random_augmentation


class NMRGuidedSampler:
    """Euler-Maruyama sampler with NMR restraint guidance"""
    
    def __init__(
        self,
        num_timesteps=500,
        t_start=1e-4,
        tau=0.3,
        log_timesteps=False,
        w_cutoff=0.99,
        nmr_energy=None,
        guidance_scale=1.0,
        guidance_start_t=0.1,
        guidance_end_t=0.8,
        guidance_schedule='linear'
    ):
        """
        Args:
            num_timesteps: Number of sampling steps
            t_start: Starting timestep
            tau: Stochasticity parameter
            log_timesteps: Use logarithmic timestep spacing
            w_cutoff: Cutoff for diffusion coefficient
            nmr_energy: NMRGuidanceEnergy object
            guidance_scale: λ in the guidance term (strength)
            guidance_start_t: Start applying guidance at this timestep
            guidance_end_t: Stop applying guidance at this timestep
            guidance_schedule: 'linear', 'cosine', or 'constant'
        """
        self.num_timesteps = num_timesteps
        self.log_timesteps = log_timesteps
        self.t_start = t_start
        self.tau = tau
        self.w_cutoff = w_cutoff
        
        # NMR guidance parameters
        self.nmr_energy = nmr_energy
        self.guidance_scale = guidance_scale
        self.guidance_start_t = guidance_start_t
        self.guidance_end_t = guidance_end_t
        self.guidance_schedule = guidance_schedule
        
        # Setup timesteps
        if self.log_timesteps:
            t = 1.0 - torch.logspace(-2, 0, self.num_timesteps + 1).flip(0)
            t = t - torch.min(t)
            t = t / torch.max(t)
            self.steps = t.clamp(min=self.t_start, max=1.0)
        else:
            self.steps = torch.linspace(
                self.t_start, 1.0, steps=self.num_timesteps + 1
            )
    
    def set_nmr_energy(self, nmr_energy):
        """Set or update NMR energy computer"""
        self.nmr_energy = nmr_energy
    
    def _guidance_weight(self, t: float) -> float:
        """
        Time-dependent guidance weight with different schedules.
        
        Returns value between 0 and guidance_scale.
        """
        if t < self.guidance_start_t or t > self.guidance_end_t:
            return 0.0
        
        # Normalized progress in guidance window
        progress = (t - self.guidance_start_t) / (self.guidance_end_t - self.guidance_start_t)
        
        if self.guidance_schedule == 'linear':
            weight = progress
        elif self.guidance_schedule == 'cosine':
            #weight = 0.5 * (1 - torch.cos(torch.tensor(progress * 3.14159)).item())
            weight = 0.5 * (1 - math.cos(progress * math.pi))
        elif self.guidance_schedule == 'constant':
            weight = 1.0
        else:
            weight = progress
        
        return self.guidance_scale * weight
    
    def diffusion_coefficient(self, t, eps=0.01):
        """Determine diffusion coefficient"""
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
        """
        Single Euler-Maruyama step with NMR guidance.
        
        Implements:
        dy = v_θ dt + λ∇log p_NMR dt + (1/2)w(t)s_θ dt + √(τ·w(t))dW
        """
        dt = t_next - t
        eps = torch.randn_like(y).to(y)
        
        # Center coordinates (SimpleFold convention)
        y = center_random_augmentation(
            y,
            batch["atom_pad_mask"],
            augmentation=False,
            centering=True,
        )
        
        # Get model prediction (velocity field)
        batched_t = repeat(t, " -> b", b=y.shape[0])
        velocity = model_fn(
            noised_pos=y,
            t=batched_t,
            feats=batch,
        )['predict_velocity']
        
        # Compute score from velocity
        score = flow.compute_score_from_velocity(velocity, y, t)
        
        # Diffusion coefficient
        diff_coeff = self.diffusion_coefficient(t)
        
        # === NMR GUIDANCE TERM ===
        guidance_weight = self._guidance_weight(t)
        
        if self.nmr_energy is not None and guidance_weight > 0:
            # Scale coordinates from SimpleFold's [-1, 1] range to Angstroms
            # SimpleFold uses scale=16.0, so coords are in range [-1/16, 1/16] nm = [-0.625, 0.625] Angstroms
            #coords_angstrom = y * 16.0 * 10.0  # Convert to Angstroms
            coords_angstrom = y * 16.0  # Convert to Angstroms

            #====================Debug Prints=====================================
            # PRINT 1: Coordinate ranges
            if abs(t - 0.5) < 0.02:  # Print around middle of sampling
                print(f"\n{'='*60}")
                print(f"DEBUG at t={t:.4f}")
                print(f"{'='*60}")
                print(f"Model coords range: [{y.min():.4f}, {y.max():.4f}]")
                print(f"Scaled coords range: [{coords_angstrom.min():.2f}, {coords_angstrom.max():.2f}] Å")
                print(f"Expected: ~[-20, +20] Å for correct scaling")
            #======================================================
            
            # Compute NMR energy gradient (in Angstroms)
            with torch.enable_grad():
                nmr_grad = self.nmr_energy.compute_gradient(coords_angstrom)
            
            # Scale gradient back to model space
            #nmr_grad = nmr_grad / (16.0 * 10.0)
            nmr_grad = nmr_grad / 16.0

            #==================== Debug prints =======================
            if abs(t - 0.5) < 0.02:
                velocity_norm = torch.norm(velocity).item()
                nmr_grad_norm = torch.norm(nmr_grad).item()
                score_norm = torch.norm(score).item()
                
                print(f"\nVector Magnitudes:")
                print(f"  Velocity norm:     {velocity_norm:.8f}")
                print(f"  NMR grad norm:     {nmr_grad_norm:.8f}")
                print(f"  Score norm:        {score_norm:.8f}")
                print(f"  Guidance weight:   {guidance_weight:.4f}")
                print(f"  Effective NMR:     {(guidance_weight * nmr_grad_norm):.8f}")
                
                # Ratio check
                ratio = (guidance_weight * nmr_grad_norm) / velocity_norm
                print(f"\nGuidance/Velocity ratio: {ratio:.6f}")
                if ratio < 0.001:
                    print("  ⚠️  WARNING: Guidance is 1000× weaker than velocity!")
                elif ratio < 0.01:
                    print("  ⚠️  WARNING: Guidance is 100× weaker than velocity!")
                elif 0.01 < ratio < 0.5:
                    print("  ✓ Guidance has meaningful influence")
                else:
                    print("  ⚠️  Guidance may be too strong!")
            #========================================================
            
            # Guidance term: -λ∇E_NMR (negative gradient for minimization)
            guidance_term = -guidance_weight * nmr_grad
        else:
            guidance_term = torch.zeros_like(y)

            #==================== Debug prints =======================
            if abs(t - 0.5) < 0.02:
                print(f"\n{'='*60}")
                print(f"DEBUG at t={t:.4f}")
                print(f"{'='*60}")
                print("❌ No guidance applied (outside window or no NMR data)")
            #=========================================================
        
        # === INTEGRATION STEP ===
        # Drift: v_θ + λ∇log p_NMR + (1/2)w(t)s_θ
        drift = velocity + guidance_term + 0.5 * diff_coeff * score
        
        # Stochastic term: √(2·τ·w(t))·dW
        if self.tau > 0 and t > self.t_start:
            stochastic_term = torch.sqrt(2.0 * self.tau * diff_coeff * abs(dt)) * eps
        else:
            stochastic_term = torch.zeros_like(y)
        
        # Update
        y_next = y + drift * dt + stochastic_term
        
        return y_next
    
    @torch.no_grad()
    def sample(self, model_fn, flow, noise, batch):
        """
        Full sampling loop with NMR guidance.
        
        Args:
            model_fn: SimpleFold model
            flow: Flow object
            noise: Initial noise
            batch: Batch dictionary
            
        Returns:
            dict with 'denoised_coords'
        """
        y_sampled = noise
        steps = self.steps.to(noise.device)
        
        for i in tqdm(
            range(self.num_timesteps),
            desc="Sampling with NMR guidance",
            total=self.num_timesteps,
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
            )
        
        return {
            "denoised_coords": y_sampled
        }
    
    @torch.no_grad()
    def sample_with_trajectory(self, model_fn, flow, noise, batch, stride=1):
        """
        Full sampling loop with NMR guidance that saves trajectory.
        
        Args:
            model_fn: SimpleFold model
            flow: Flow object
            noise: Initial noise
            batch: Batch dictionary
            stride: Save every N-th frame
            
        Returns:
            dict with 'denoised_coords'
            list: Trajectory of coordinates [timesteps, batch, n_atoms, 3]
        """
        y_sampled = noise
        steps = self.steps.to(noise.device)
        
        # Store trajectory - save initial state
        trajectory = []
        trajectory.append(y_sampled.clone().cpu())
        
        for i in tqdm(
            range(self.num_timesteps),
            desc="Sampling with NMR guidance (saving trajectory)",
            total=self.num_timesteps,
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
            )
            
            # Save trajectory at specified stride
            if (i + 1) % stride == 0:
                trajectory.append(y_sampled.clone().cpu())
        
        return {
            "denoised_coords": y_sampled
        }, trajectory