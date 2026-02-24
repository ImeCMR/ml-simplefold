"""
Modified SimpleFold sampler with NMR guidance and MELD-style dynamic restraint selection.
Extends the EMSampler with inference-time NMR restraint guidance.
"""
import torch
import math
from tqdm import tqdm
from einops import repeat
from typing import Dict, Optional
from utils.boltz_utils import center_random_augmentation


class NMRGuidedSampler:
    """Euler-Maruyama sampler with MELD-style dynamic restraint selection"""
    
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
        guidance_schedule='linear',
        reselect_every=1  # How often to reselect active groups
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
            reselect_every: How often to reselect active groups (MELD)
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
        
        # MELD dynamic selection
        self.reselect_every = reselect_every
        self.step_counter = 0
        self.cached_distance_groups = None
        self.cached_torsion_groups = None
        
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
        Single Euler-Maruyama step with MELD dynamic group selection.
        
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
        
        # === MELD GUIDANCE ===
        guidance_weight = self._guidance_weight(t)


        #================================ debug prints =====================
        # **PRINT 1: Check if guidance is active**
        if self.step_counter % 50 == 0:  # Print every 50 steps to avoid spam
            print(f"\n📊 Step {self.step_counter}, t={t:.4f}")
            print(f"  Guidance weight: {guidance_weight:.6f}")
            print(f"  Diffusion coeff: {diff_coeff:.6f}")
        #====================================================================
        
        if self.nmr_energy is not None and guidance_weight > 0:
            # Scale coordinates to Angstroms
            coords_angstrom = y * 16.0
            
            # === MELD: Re-select active groups periodically ===
            if self.step_counter % self.reselect_every == 0:
                print(f"\n🔄 Step {self.step_counter}: Re-selecting active restraints...")
                
                # Select best distance groups
                active_distance = self.nmr_energy.select_active_groups_meld(
                    coords_angstrom,
                    restraint_type='distance'
                )
                
                # Select best torsion groups
                active_torsion = self.nmr_energy.select_active_groups_meld(
                    coords_angstrom,
                    restraint_type='torsion'
                )
                
                self.cached_distance_groups = active_distance
                self.cached_torsion_groups = active_torsion
                
                print(f"  ✓ Active distance groups: {len(active_distance)}/{len(self.nmr_energy.distance_groups)}")
                print(f"  ✓ Active torsion groups: {len(active_torsion)}/{len(self.nmr_energy.torsion_groups)}")
            else:
                # Use previously selected groups
                active_distance = self.cached_distance_groups or list(range(self.nmr_energy.n_distance_active))
                active_torsion = self.cached_torsion_groups or list(range(self.nmr_energy.n_torsion_active))
            
            # Compute gradient with selected groups
            with torch.enable_grad():
                coords_copy = coords_angstrom.clone().detach().requires_grad_(True)
                
                # Compute energies with selected active groups
                dist_energy = self.nmr_energy.compute_distance_energy(
                    coords_copy,
                    active_groups=active_distance
                )
                tors_energy = self.nmr_energy.compute_torsion_energy(
                    coords_copy,
                    active_groups=active_torsion
                )
                
                total_energy = dist_energy + tors_energy

                #============================ debug prints ============================
                if self.step_counter % 50 == 0:
                    print(f"  💡 NMR Energies:")
                    print(f"    Distance: {dist_energy.item():.2f} kJ/mol")
                    print(f"    Torsion: {tors_energy.item():.2f} kJ/mol")
                    print(f"    Total: {total_energy.item():.2f} kJ/mol")
                #=====================================================================

                    
                nmr_grad = torch.autograd.grad(total_energy.sum(), coords_copy)[0]
            
            # Scale gradient back to model space
            nmr_grad = nmr_grad / 16.0
            
            # Guidance term: -λ∇E_NMR (negative gradient for minimization)
            guidance_term = -guidance_weight * nmr_grad
            
            #============================== debug prints ==============================
            if self.step_counter % 50 == 0:
                print(f"  📈 Gradient Analysis:")
                print(f"    NMR grad norm: {torch.norm(nmr_grad).item():.8f}")
                print(f"    Guidance term norm: {torch.norm(guidance_term).item():.8f}")
                print(f"    Velocity norm: {torch.norm(velocity).item():.8f}")
                print(f"    Score norm: {torch.norm(score).item():.8f}")
            #==========================================================================
            # Increment step counter
            self.step_counter += 1
        else:
            guidance_term = torch.zeros_like(y)

            #============================== debug prints ==============================
            if self.step_counter % 100 == 0 and self.nmr_energy is not None:
                print(f"  ⚠️  Step {self.step_counter}: Guidance inactive (weight={guidance_weight:.6f})")
            #=========================================================================
        
        # === INTEGRATION STEP ===
        # Drift: v_θ + λ∇log p_NMR + (1/2)w(t)s_θ
        drift = velocity + guidance_term + 0.5 * diff_coeff * score

        #================================ debug prints =====================
        if self.step_counter % 50 == 0:
            print(f"  🔧 Integration:")
            print(f"    Drift norm: {torch.norm(drift).item():.8f}")
            print(f"    dt: {dt:.6f}")
            print(f"    Update magnitude: {torch.norm(drift * dt).item():.8f}")
        #===================================================================
        
        # Stochastic term: √(2·τ·w(t))·dW
        if self.tau > 0 and t > self.t_start:
            stochastic_term = torch.sqrt(2.0 * self.tau * diff_coeff * abs(dt)) * eps
        else:
            stochastic_term = torch.zeros_like(y)
        
        # Update
        y_next = y + drift * dt + stochastic_term
        
        #===================================== debug prints ============================
        # **PRINT 6: Coordinate changes**
        if self.step_counter % 50 == 0:
            coord_change = torch.norm(y_next - y).item()
            print(f"  📍 Coordinate change: {coord_change:.6f}")
            print(f"  {'='*60}")
        #==============================================================================

        return y_next
    
    @torch.no_grad()
    def sample(self, model_fn, flow, noise, batch):
        """
        Full sampling loop with MELD-style NMR guidance.
        
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
            desc="Sampling with MELD-NMR guidance",
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
        Full sampling loop with MELD-style NMR guidance that saves trajectory.
        
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
            desc="Sampling with MELD-NMR guidance (saving trajectory)",
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