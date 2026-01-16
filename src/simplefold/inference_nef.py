#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
NEF-Guided Inference Module for SimpleNOEFold

This module provides the high-level interface for structure prediction
guided by NMR NOESY data encoded in NEF files.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Tuple
from copy import deepcopy

from nef.parser import NEFParser, DistanceRestraint
from energy.noe_energy import NOEEnergy, NOEDistance
from energy.geometric_prior import GeometricPrior
from model.flow import LinearPath
from model.torch.steering import BayesianSteeredEMSampler
from processor.protein_processor import ProteinDataProcessor
from utils.esm_utils import _af2_to_esm, esm_registry


class BayesianEnergyGradient(nn.Module):
    """
    Computes the steering force as the negative gradient of total Bayesian energy.
    
    F_steering = -∇_{x_t} [E_NOE(x_t) + E_Geom(x_t)]
    """
    
    def __init__(
        self,
        noe_energy: NOEEnergy,
        geometric_prior: GeometricPrior,
        noe_weight: float = 1.0,
        geom_weight: float = 1.0,
    ):
        """
        Parameters
        ----------
        noe_energy : NOEEnergy
            NOE energy function
        geometric_prior : GeometricPrior
            Geometric prior function
        noe_weight : float
            Weight for NOE energy term
        geom_weight : float
            Weight for geometric energy term
        """
        super().__init__()
        self.noe_energy = noe_energy
        self.geometric_prior = geometric_prior
        self.noe_weight = noe_weight
        self.geom_weight = geom_weight
    
    def forward(
        self,
        coords: torch.Tensor,
        t: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute steering force and total energy.
        
        Parameters
        ----------
        coords : torch.Tensor
            Atomic coordinates [B, N_atoms, 3], requires_grad=True
        t : torch.Tensor
            Time step [B]
        atom_mask : torch.Tensor
            Valid atom mask [B, N_atoms]
        
        Returns
        -------
        Tuple[torch.Tensor, Dict]
            - Steering force [B, N_atoms, 3]
            - Statistics dictionary
        """
        # Ensure coordinates require gradients
        coords_opt = coords.detach().clone().requires_grad_(True)
        
        # Compute NOE energy
        noe_energy, noe_stats = self.noe_energy(coords_opt, t)
        noe_energy = torch.sum(noe_energy * self.noe_weight)
        
        # Compute geometric energy
        geom_energy, geom_stats = self.geometric_prior(coords_opt, atom_mask)
        geom_energy = torch.sum(geom_energy * self.geom_weight)
        
        # Total energy
        total_energy = noe_energy + geom_energy
        
        # Compute gradient (steering force)
        total_energy.backward()
        steering_force = -coords_opt.grad  # [B, N_atoms, 3]
        
        # Combine statistics
        stats = {
            "noe_energy": noe_energy.detach(),
            "geom_energy": geom_energy.detach(),
            **{f"noe/{k}": v for k, v in noe_stats.items()},
            **{f"geom/{k}": v for k, v in geom_stats.items()},
        }
        
        return steering_force, stats


class SimpleNOEFoldInference:
    """
    High-level inference interface for NEF-guided structure prediction.
    
    Workflow:
    1. Load pretrained SimpleFold model
    2. Parse NEF file to extract NOESY distance restraints
    3. Create Bayesian energy functions (NOE + Geometric)
    4. Use steered sampler to generate structure conditioned on NMR data
    """
    
    def __init__(
        self,
        model_checkpoint: Path,
        device: str = "cuda",
        esm_model: str = "esm2_3B",
    ):
        """
        Parameters
        ----------
        model_checkpoint : Path
            Path to pretrained SimpleFold checkpoint
        device : str
            Device to use (cuda or cpu)
        esm_model : str
            ESM model variant
        """
        self.device = torch.device(device)
        self.model_checkpoint = Path(model_checkpoint)
        
        # Load ESM model
        self.esm_model, self.esm_dict = esm_registry[esm_model]()
        self.esm_model.eval()
        self.af2_to_esm = _af2_to_esm(self.esm_dict)
        self.esm_model = self.esm_model.to(self.device)
        self.af2_to_esm = self.af2_to_esm.to(self.device)
        
        # Initialize processor
        self.processor = ProteinDataProcessor(device=self.device)
        
        # Placeholder for model (will be loaded later)
        self.model = None
        self.model_ema = None
    
    def load_model(self, model):
        """Set the pretrained SimpleFold model."""
        self.model = model
        self.model_ema = model  # Use EMA version if available
    
    def parse_nef_file(
        self,
        nef_file: Path,
        atom_mapping: Optional[Dict] = None,
    ) -> Tuple[NOEEnergy, Dict]:
        """
        Parse NEF file and create NOE energy function.
        
        Parameters
        ----------
        nef_file : Path
            Path to NEF file
        atom_mapping : Optional[Dict]
            Custom mapping from NEF atom names to structure indices
        
        Returns
        -------
        Tuple[NOEEnergy, Dict]
            - NOE energy function
            - Statistics from NEF file
        """
        # Parse NEF
        parser = NEFParser(nef_file)
        restraints_text = parser.get_restraints()
        
        print(parser.summary())
        
        # Convert text restraints to NOEDistance objects
        # This requires mapping residue numbers and atom names to coordinate indices
        noe_distances = []
        
        for restraint in restraints_text:
            # Simple mapping: assume single chain with sequential residues
            # In production, you'd use a proper PDB parser
            res1_idx = restraint.atom1.sequence_code - 1  # 0-indexed
            res2_idx = restraint.atom2.sequence_code - 1
            
            # Map atom names to standard indices
            atom1_idx = self._get_atom_index(res1_idx, restraint.atom1.atom_name)
            atom2_idx = self._get_atom_index(res2_idx, restraint.atom2.atom_name)
            
            if atom1_idx is not None and atom2_idx is not None:
                noe_dist = NOEDistance(
                    atom_indices=[(atom1_idx, atom2_idx)],
                    upper_bound=restraint.upper_bound,
                    lower_bound=restraint.lower_bound,
                    weight=restraint.weight,
                )
                noe_distances.append(noe_dist)
        
        # Create NOE energy function
        noe_energy = NOEEnergy(
            restraints=noe_distances,
            potential_type="gaussian",
            base_sigma=0.5,
            time_dependent=True,
        )
        
        stats = {
            "num_restraints": len(noe_distances),
            "nef_file": str(nef_file),
        }
        
        return noe_energy, stats
    
    def _get_atom_index(self, residue_idx: int, atom_name: str) -> Optional[int]:
        """
        Map residue index and atom name to coordinate array index.
        
        This is a simplified version. In production, use proper PDB parsing.
        """
        # Simplified mapping: assume standard atom order
        # CA, CB, C, N, O per residue
        atom_order = {
            'CA': 0,
            'CB': 1,
            'C': 2,
            'N': 3,
            'O': 4,
        }
        
        if atom_name not in atom_order:
            return None
        
        return residue_idx * 5 + atom_order[atom_name]
    
    @torch.no_grad()
    def predict(
        self,
        sequence: str,
        nef_file: Path,
        num_samples: int = 5,
        steering_weight: float = 1.0,
        steering_start_t: float = 0.5,
    ) -> Dict:
        """
        Predict protein structure guided by NMR data.
        
        Parameters
        ----------
        sequence : str
            Protein sequence (standard amino acid codes)
        nef_file : Path
            Path to NEF file with distance restraints
        num_samples : int
            Number of samples to generate
        steering_weight : float
            Maximum steering force weight (γ_max)
        steering_start_t : float
            Time to start steering (0 to 1)
        
        Returns
        -------
        Dict
            Dictionary with:
            - "coords": Generated coordinates [num_samples, N_atoms, 3]
            - "energy": Bayesian energy for each sample
            - "violations": Distance violations for each sample
        """
        # Parse NEF file
        noe_energy, nef_stats = self.parse_nef_file(nef_file)
        
        # Create geometric prior
        geometric_prior = GeometricPrior(
            clash_cutoff=2.5,
            use_clash=True,
            use_covalent=True,
        )
        
        # Create combined energy gradient
        energy_gradient = BayesianEnergyGradient(
            noe_energy=noe_energy,
            geometric_prior=geometric_prior,
            noe_weight=1.0,
            geom_weight=0.1,
        )
        
        # Create steered sampler
        sampler = BayesianSteeredEMSampler(
            num_timesteps=500,
            steering_schedule_type="linear",
            steering_start_t=steering_start_t,
            steering_max_weight=steering_weight,
        )
        
        # Prepare batch
        # Note: This is simplified. In production, use actual data loading
        batch = self._prepare_batch(sequence)
        
        # Define steering function
        def steering_fn(coords: torch.Tensor, t: torch.Tensor):
            force, stats = energy_gradient(
                coords,
                t,
                batch["atom_pad_mask"],
            )
            return force
        
        # Sample structures
        results = []
        for i in range(num_samples):
            noise = torch.randn_like(batch['coords']).to(self.device)
            
            out_dict = sampler.sample(
                self.model_ema.forward,
                LinearPath(),  # Flow path
                noise,
                batch,
                steering_fn=steering_fn,
            )
            
            results.append(out_dict)
        
        return {
            "coords": torch.cat([r["denoised_coords"] for r in results], dim=0),
            "nef_stats": nef_stats,
            "sequence": sequence,
        }
    
    def _prepare_batch(self, sequence: str) -> Dict:
        """Prepare batch data from sequence."""
        # This is simplified. In production, use proper data loading
        batch_size = 1
        n_atoms = len(sequence) * 5  # Simplified: 5 atoms per residue
        
        batch = {
            "coords": torch.randn(batch_size, n_atoms, 3),
            "atom_pad_mask": torch.ones(batch_size, n_atoms),
            "sequence": sequence,
        }
        
        return batch
