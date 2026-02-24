"""
NMR restraint parsing and energy computation for SimpleFold guidance.
Compatible with MELD's NOE.dat and rotamers.dat formats.
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from boltz_data_pipeline import const


@dataclass
class DistanceRestraint:
    """NOE distance restraint"""
    res_i: int
    atom_i: str
    res_j: int
    atom_j: str
    distance: float  # in Angstroms
    k: float = 350.0  # force constant kJ/mol/nm^2
    
    @property
    def r3(self):
        """Target distance in nm"""
        #return self.distance / 10.0
        return (self.distance - 0.5) / 10.0
    
    @property
    def r4(self):
        """Upper tolerance in nm"""
        #return self.r3 + 0.2
        return (self.distance + 0.5) / 10.0


@dataclass
class TorsionRestraint:
    """TALOS torsion angle restraint"""
    res1: int
    atom1: str
    res2: int
    atom2: str
    res3: int
    atom3: str
    res4: int
    atom4: str
    phi_min: float  # degrees
    phi_max: float  # degrees
    k: float = 0.05  # force constant
    
    @property
    def phi_mean(self):
        return (self.phi_min + self.phi_max) / 2.0
    
    @property
    def phi_std(self):
        return abs(self.phi_max - self.phi_min) / 2.0


class NMRRestraintParser:
    """Parse MELD-format NMR data files"""
    
    @staticmethod
    def parse_noe_file(filepath: str) -> List[List[DistanceRestraint]]:
        """
        Parse NOE.dat file (MELD format).
        Groups separated by empty lines.
        
        Format per line:
        res_i atom_i res_j atom_j distance(Å)
        """
        restraint_groups = []
        current_group = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_group:
                        restraint_groups.append(current_group)
                        current_group = []
                    continue
                
                parts = line.split()
                if len(parts) >= 5:
                    restraint = DistanceRestraint(
                        res_i=int(parts[0]) - 1,  # Convert to 0-indexed
                        atom_i=parts[1],
                        res_j=int(parts[2]) - 1,
                        atom_j=parts[3],
                        distance=float(parts[4]),  # Keep in Angstroms
                    )
                    current_group.append(restraint)
        
        if current_group:
            restraint_groups.append(current_group)
        
        print(f"Loaded {len(restraint_groups)} NOE restraint groups from {filepath}")
        return restraint_groups
    
    @staticmethod
    def parse_talos_file(filepath: str) -> List[List[TorsionRestraint]]:
        """
        Parse rotamers.dat file (MELD format).
        Groups separated by empty lines.
        
        Format per line:
        res1 atom1 res2 atom2 res3 atom3 res4 atom4 phi_min phi_max
        """
        restraint_groups = []
        current_group = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_group:
                        restraint_groups.append(current_group)
                        current_group = []
                    continue
                
                parts = line.split()
                if len(parts) >= 10:
                    restraint = TorsionRestraint(
                        res1=int(parts[0]) - 1,  # Convert to 0-indexed
                        atom1=parts[1],
                        res2=int(parts[2]) - 1,
                        atom2=parts[3],
                        res3=int(parts[4]) - 1,
                        atom3=parts[5],
                        res4=int(parts[6]) - 1,
                        atom4=parts[7],
                        phi_min=float(parts[8]),
                        phi_max=float(parts[9]),
                    )
                    current_group.append(restraint)
        
        if current_group:
            restraint_groups.append(current_group)
        
        print(f"Loaded {len(restraint_groups)} torsion restraint groups from {filepath}")
        return restraint_groups


class NMRGuidanceEnergy:
    """Compute NMR restraint energies and gradients for SimpleFold"""
    
    def __init__(
        self,
        distance_groups: List[List[DistanceRestraint]],
        torsion_groups: List[List[TorsionRestraint]],
        batch: Dict,
        n_distance_active: Optional[int] = None,
        n_torsion_active: Optional[int] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            distance_groups: List of NOE restraint groups
            torsion_groups: List of torsion restraint groups
            batch: SimpleFold batch dict
            n_distance_active: Number of distance groups to activate (None = all)
            n_torsion_active: Number of torsion groups to activate (None = all)
        """
        self.distance_groups = distance_groups
        self.torsion_groups = torsion_groups
        self.batch = batch
        self.device = device
        
        # MELD-style selective activation
        self.n_distance_active = n_distance_active or len(distance_groups)
        self.n_torsion_active = n_torsion_active or len(torsion_groups)
        
        # Build atom index maps from SimpleFold batch
        self._build_index_maps()

    #=========================== New MELD features =========================
    def select_active_groups_meld(
        self,
        coords: torch.Tensor,
        restraint_type: str = 'distance'
    ) -> List[int]:
        """
        MELD selection: Choose top N% most satisfied restraint groups.
        
        For each group:
        1. Compute minimum violation across all ambiguous assignments (OR logic)
        2. Rank groups by this minimum violation
        3. Activate only the best N groups
        
        Returns:
            List of group indices to activate
        """
        if restraint_type == 'distance':
            groups = self.distance_groups
            n_active = self.n_distance_active
        else:
            groups = self.torsion_groups
            n_active = self.n_torsion_active
        
        if not groups:
            return []
        
        #=========================== Debug prints ============================
        print(f"\n  🎯 MELD Selection ({restraint_type}):")
        print(f"    Total groups: {len(groups)}")
        print(f"    Selecting: {n_active}")
        #=====================================================================

        # Compute satisfaction score for each group
        group_scores = []
        
        for group_idx, group in enumerate(groups):
            min_violation = float('inf')
            
            for restraint in group:
                # Compute violation for this assignment
                if restraint_type == 'distance':
                    violation = self._compute_distance_violation(coords, restraint)
                else:
                    violation = self._compute_torsion_violation(coords, restraint)
                
                # Keep minimum (OR logic within group)
                min_violation = min(min_violation, violation)
            
            # Lower violation = better satisfaction = higher priority
            group_scores.append((group_idx, min_violation))
        
        # Sort by violation (ascending = most satisfied first)
        group_scores.sort(key=lambda x: x[1])

        #========================================== Debug prints ============================
        print(f"    Best 3 groups: {[(idx, f'{viol:.4f}') for idx, viol in group_scores[:3]]}")
        print(f"    Worst 3 groups: {[(idx, f'{viol:.4f}') for idx, viol in group_scores[-3:]]}")
        #======================================================================================
        
        # Return top N most satisfied groups
        active_groups = [idx for idx, _ in group_scores[:n_active]]
        
        return active_groups

    def _compute_distance_violation(
        self, 
        coords: torch.Tensor, 
        restraint: DistanceRestraint
    ) -> float:
        """Compute violation for a single distance restraint"""
        idx_i = self._get_atom_index(restraint.res_i, restraint.atom_i)
        idx_j = self._get_atom_index(restraint.res_j, restraint.atom_j)
        
        if idx_i is None or idx_j is None:
            return float('inf')
        
        pos_i = coords[0, idx_i, :]  # Take first batch element
        pos_j = coords[0, idx_j, :]
        dist = torch.norm(pos_j - pos_i).item()
        dist_nm = dist / 10.0
        
        # Flat-bottom violation
        violation = max(0, dist_nm - restraint.r4)
        
        return violation
    
    def _compute_torsion_violation(
        self,
        coords: torch.Tensor,
        restraint: TorsionRestraint
    ) -> float:
        """Compute violation for a single torsion restraint"""
        idx1 = self._get_atom_index(restraint.res1, restraint.atom1)
        idx2 = self._get_atom_index(restraint.res2, restraint.atom2)
        idx3 = self._get_atom_index(restraint.res3, restraint.atom3)
        idx4 = self._get_atom_index(restraint.res4, restraint.atom4)
        
        if None in [idx1, idx2, idx3, idx4]:
            return float('inf')
        
        # Compute dihedral
        phi = self._compute_dihedral(
            coords[0, idx1, :],
            coords[0, idx2, :],
            coords[0, idx3, :],
            coords[0, idx4, :]
        )
        phi_deg = torch.rad2deg(phi).item()
        
        # Check if within bounds
        if restraint.phi_min <= phi_deg <= restraint.phi_max:
            return 0.0
        
        # Compute angular distance to nearest bound
        dist_to_min = abs(phi_deg - restraint.phi_min)
        dist_to_max = abs(phi_deg - restraint.phi_max)
        
        return min(dist_to_min, dist_to_max)
    
    #=======================================================================
        
#    def _build_index_maps(self):
#        """
#        Map (residue_idx, atom_name) to global atom indices.
#        
#        SimpleFold uses:
#        - batch["ref_atom_name_chars"]: [batch_size, N_atoms, 4, 64] one-hot encoded atom names
#        - batch["ref_space_uid"]: [batch_size, N_atoms] residue indices for each atom
#        """
#        self.atom_index_map = {}
#        
#        # Decode atom names from one-hot encoding
#        ref_atom_names = self.batch["ref_atom_name_chars"]  # [batch_size, N_atoms, 4, 64]
#        ref_space_uid = self.batch["ref_space_uid"]  # [batch_size, N_atoms]
#        
#        # === FIX: Handle batch dimension ===
#        # SimpleFold batches have shape [batch_size, ...], take first example
#        if len(ref_atom_names.shape) == 4:  # Has batch dimension
#            ref_atom_names = ref_atom_names[0]  # Now [N_atoms, 4, 64]
#            ref_space_uid = ref_space_uid[0]    # Now [N_atoms]
#        
#        n_atoms = ref_atom_names.shape[0]
#        
#        # Character mapping (A-Z, 0-9, etc.)
#        chars = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#        
#        for atom_idx in range(n_atoms):
#            res_idx = int(ref_space_uid[atom_idx].item())  # Now this works!
#            
#            # Decode atom name from one-hot
#            atom_name_chars = []
#            for char_pos in range(4):
#                char_idx = torch.argmax(ref_atom_names[atom_idx, char_pos]).item()
#                if char_idx < len(chars):
#                    atom_name_chars.append(chars[char_idx])
#            
#            atom_name = ''.join(atom_name_chars).strip()
#            
#            # Map (res_idx, atom_name) -> atom_idx
#            self.atom_index_map[(res_idx, atom_name)] = atom_idx
#        
#        print(f"Built atom index map with {len(self.atom_index_map)} entries")
#
#        #============================= Debug prints ==============================
#        print("\nFirst 20 atoms in map:")
#        for i, (key, val) in enumerate(list(self.atom_index_map.items())[:20]):
#            print(f"  ({key[0]:3d}, '{key[1]:4s}') -> atom {val}")
#        
#        print("\nSample CA atoms:")
#        ca_atoms = [(k, v) for k, v in self.atom_index_map.items() if k[1] == 'CA']
#        for i, (key, val) in enumerate(ca_atoms[:10]):
#            print(f"  Residue {key[0]} CA -> atom {val}")
#        #=========================================================================


#**************************** changed _build_index_maps ====================
#    def _build_index_maps(self):
#        """
#        Map (residue_idx, atom_name) to global atom indices.
#        Use SimpleFold's ref_atoms structure with proper residue type detection.
#        """
#        self.atom_index_map = {}
#
#        #=============================== Debug prints ==============================
#        # DEBUG: Print all available batch keys
#        print(f"\n{'='*60}")
#        print("DEBUG: Available batch keys:")
#        print(f"{'='*60}")
#        for key in self.batch.keys():
#            if isinstance(self.batch[key], torch.Tensor):
#                print(f"  {key}: shape {self.batch[key].shape}, dtype {self.batch[key].dtype}")
#            else:
#                print(f"  {key}: type {type(self.batch[key])}")
#        print(f"{'='*60}\n")
#        #==========================================================================
#        
#        # SimpleFold's atom ordering from const.py [2]
#        ref_atoms = {
#            "ALA": ["N", "CA", "C", "O", "CB"],
#            "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
#            "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
#            "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
#            "CYS": ["N", "CA", "C", "O", "CB", "SG"],
#            "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
#            "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
#            "GLY": ["N", "CA", "C", "O"],
#            "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
#            "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
#            "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
#            "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
#            "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
#            "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
#            "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
#            "SER": ["N", "CA", "C", "O", "CB", "OG"],
#            "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
#            "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
#            "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
#            "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
#        }
#        
#        # Get ref_space_uid (which residue each atom belongs to)
#        ref_space_uid = self.batch["ref_space_uid"]
#        if len(ref_space_uid.shape) == 2:
#            ref_space_uid = ref_space_uid[0]
#        
#        # === FIX: Get residue types from token field instead of restype ===
#        # SimpleFold uses "token" field which has indices into the token list
#        token = self.batch.get("token")
#        if token is None:
#            print("ERROR: 'token' field not found in batch!")
#            return
#        
#        if len(token.shape) == 2:
#            token = token[0]
#        
#        # Token index to 3-letter code mapping (from const.py tokens list)
#        token_to_aa = [
#            None,  # 0: <pad>
#            None,  # 1: -
#            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
#            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
#            "UNK",  # 22: unknown protein
#        ]
#        
#        # Build residue type map: res_idx -> 3-letter code
#        residue_types = {}
#        for res_idx in range(len(token)):
#            token_idx = int(token[res_idx].item())
#            if token_idx < len(token_to_aa) and token_to_aa[token_idx] is not None:
#                residue_types[res_idx] = token_to_aa[token_idx]
#        
#        print(f"DEBUG: Found {len(residue_types)} residue types")
#        print(f"Sample residue types: {list(residue_types.items())[:10]}")
#        
#        # Build atom index map by iterating through all atoms
#        residue_atom_counts = {}
#        
#        for atom_idx in range(len(ref_space_uid)):
#            res_idx = int(ref_space_uid[atom_idx].item())
#            
#            # Get residue type
#            res_type = residue_types.get(res_idx)
#            if res_type is None:
#                continue
#            
#            # Get expected atoms for this residue type
#            expected_atoms = ref_atoms.get(res_type, [])
#            if not expected_atoms:
#                continue
#            
#            # Track how many atoms of this residue we've seen
#            if res_idx not in residue_atom_counts:
#                residue_atom_counts[res_idx] = 0
#            
#            atom_position = residue_atom_counts[res_idx]
#            
#            # Map to the corresponding atom name
#            if atom_position < len(expected_atoms):
#                atom_name = expected_atoms[atom_position]
#                self.atom_index_map[(res_idx, atom_name)] = atom_idx
#                residue_atom_counts[res_idx] += 1
#        
#        print(f"Built atom index map with {len(self.atom_index_map)} entries")
#        
#        # Debug prints
#        print("\nFirst 20 atoms in map:")
#        for i, (key, val) in enumerate(list(self.atom_index_map.items())[:20]):
#            print(f"  Residue {key[0]:3d}, Atom '{key[1]:4s}' -> atom {val}")
#        
#        print("\nSample CA atoms:")
#        ca_atoms = [(k, v) for k, v in self.atom_index_map.items() if k[1] == 'CA']
#        for i, (key, val) in enumerate(ca_atoms[:10]):
#            print(f"  Residue {key[0]} CA -> atom {val}")

#===============================================================================


#=========================== updated _build_index_maps ============================
    def _build_index_maps(self):
        """
        Map (residue_idx, atom_name) to global atom indices.
        Use SimpleFold's native atom data structure.
        """
        self.atom_index_map = {}
        

        #=============================== Debug prints ==============================
        print(f"\n{'='*60}")
        print("Building Atom Index Map")
        print(f"{'='*60}")
        #==========================================================================


        # Get atom data from batch
        # SimpleFold stores raw atom names in the 'atom_data' if available
        # Otherwise, reconstruct from token information
        
        # Method 1: If you have access to the original structure object
        # (passed during NMRGuidanceEnergy initialization)
        if hasattr(self, 'structure') and self.structure is not None:

            #================================ Debug prints ==============================
            print("Using Method 1: Direct structure object")
            print(f"Number of atoms in structure: {len(self.structure.atoms)}")
            #===========================================================================
            for atom_idx, atom in enumerate(self.structure.atoms):
                res_idx = int(self.batch["ref_space_uid"][0, atom_idx].item())
                
                # Decode atom name from bytes
                atom_name_bytes = atom["name"]
                atom_name = ''.join([chr(c) for c in atom_name_bytes if c != 0]).strip()
                
                if atom_name:
                    self.atom_index_map[(res_idx, atom_name)] = atom_idx
        else:

            #=============================== Debug prints ==============================
            print("Using Method 2: Token-based reconstruction")
            #==============================================================================

            # Method 2: Use token-based reconstruction
            token_data = self.batch.get("res_type")  # [batch, n_tokens, n_classes]
            ref_space_uid = self.batch["ref_space_uid"]  # [batch, n_atoms]

            #==================================== Debug prints ============================
            print(f"ref_space_uid shape: {ref_space_uid.shape}")
            if token_data is not None:
                print(f"token_data shape: {token_data.shape}")
            else:
                print("WARNING: token_data is None!")
            #====================================================================
            
            if len(ref_space_uid.shape) == 2:
                ref_space_uid = ref_space_uid[0]
            
            # Get residue types
            if token_data is not None and len(token_data.shape) == 3:
                token_indices = torch.argmax(token_data[0], dim=-1)  # [n_tokens]

                #===================================================== Debug prints ============================
                print(f"Token indices shape: {token_indices.shape}")
                print(f"Total atoms to process: {len(ref_space_uid)}")
                #===========================================================
                
                # Build atom map using SimpleFold's const.py reference
                atom_counter = {}

                #===================================== Debug prints ============================
                skipped_atoms = 0
                #==============================================================
                for atom_idx in range(len(ref_space_uid)):
                    res_idx = int(ref_space_uid[atom_idx].item())
                    
                    # Get residue type from token
                    token_idx = int(token_indices[res_idx].item())
                    if token_idx >= len(const.tokens):

                        #============================== Debug prints ==============================
                        skipped_atoms += 1
                        if skipped_atoms <= 5:
                            print(f"  WARNING: Skipping atom {atom_idx}, invalid token_idx={token_idx}")
                        #==========================================================
                        continue
                        
                    res_type = const.tokens[token_idx]
                    
                    # Get expected atom order for this residue type
                    if res_type not in const.ref_atoms:

                        #============================== Debug prints ==============================
                        skipped_atoms += 1
                        if skipped_atoms <= 5:
                            print(f"  WARNING: Unknown residue type '{res_type}' at res {res_idx}")
                        #==============================================
                        continue
                        
                    expected_atoms = const.ref_atoms[res_type]
                    
                    # Track how many atoms we've seen for this residue
                    if res_idx not in atom_counter:
                        atom_counter[res_idx] = 0
                    
                    atom_pos = atom_counter[res_idx]
                    if atom_pos < len(expected_atoms):
                        atom_name = expected_atoms[atom_pos]
                        self.atom_index_map[(res_idx, atom_name)] = atom_idx
                        atom_counter[res_idx] += 1
        #========================================= Debug prints ============================
                print(f"\nMethod 2 Summary:")
                print(f"  Total atoms processed: {len(ref_space_uid)}")
                print(f"  Successfully mapped: {len(self.atom_index_map)}")
                print(f"  Skipped atoms: {skipped_atoms}")
                print(f"  Unique residues: {len(atom_counter)}")
            else:
                print("ERROR: Cannot build atom map - invalid token_data")
                if token_data is None:
                    print("  token_data is None")
                else:
                    print(f"  token_data has wrong shape: {token_data.shape}")
        #====================================================================================
        
        #========================= Debug prints ============================
        print(f"\n{'='*60}")
        print(f"Built atom index map with {len(self.atom_index_map)} entries")
        print(f"{'='*60}\n")
        
        # Verify with backbone atoms
        print("Checking backbone atoms (first 3 residues):")
        backbone_found = 0
        for res in range(min(3, 10)):
            for atom in ['N', 'CA', 'C', 'O']:
                idx = self.atom_index_map.get((res, atom))
                if idx is not None:
                    print(f"  Residue {res:3d}, Atom '{atom:4s}' -> {idx}")

        if backbone_found == 0:
            print("  ⚠️  WARNING: No backbone atoms found! Check residue indexing.")
        else:
            print(f"  ✓ Found {backbone_found} backbone atoms")
        
        # **PRINT 14: Check first restraint atoms (if available)**
        if hasattr(self, 'distance_groups') and self.distance_groups:
            print("\nVerifying first distance restraint atoms:")
            first_restraint = self.distance_groups[0][0]
            idx_i = self.atom_index_map.get((first_restraint.res_i, first_restraint.atom_i))
            idx_j = self.atom_index_map.get((first_restraint.res_j, first_restraint.atom_j))
            print(f"  Restraint: res{first_restraint.res_i}:{first_restraint.atom_i} <-> res{first_restraint.res_j}:{first_restraint.atom_j}")
            print(f"  Mapped to: {idx_i} <-> {idx_j}")
            if idx_i is None or idx_j is None:
                print(f"  ⚠️  ERROR: Restraint atoms not found in map!")
        
        print(f"{'='*60}\n")
        #==================================================================================


    def _get_atom_index(self, res_idx: int, atom_name: str) -> Optional[int]:
        """Get global atom index from residue index and atom name"""
        return self.atom_index_map.get((res_idx, atom_name))
    
    def compute_distance_energy(
        self,
        coords: torch.Tensor,
        active_groups: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute NOE distance restraint energy with MELD OR logic.
        Only minimum violation within each group contributes.
        
        Args:
            coords: [batch, n_atoms, 3] atom coordinates in Angstroms
            active_groups: Which restraint groups to use
        
        Returns:
            energy: [batch] total energy in kJ/mol
        """
        if active_groups is None:
            active_groups = list(range(self.n_distance_active))
        
        batch_size = coords.shape[0]
        total_energy = torch.zeros(batch_size, device=coords.device, dtype=coords.dtype)
        has_contribution = False

        #==================================== Debug prints ============================
        if len(active_groups) > 0:
            print(f"\n  📏 Distance Energy Computation:")
            print(f"    Active groups: {len(active_groups)}")
            print(f"    Coords shape: {coords.shape}")
        #=============================================================================
        
        for group_idx in active_groups:
            if group_idx >= len(self.distance_groups):
                continue
            
            group = self.distance_groups[group_idx]
            group_violations = []
            
            #==================================== Debug prints ============================
            if group_idx < 3:  # Only print first 3 groups
                print(f"\n    Group {group_idx}: {len(group)} restraints")
            #========================================================================

            # Compute violation for EACH assignment in group
            for restraint in group:
                
                #=========================== Debug prints ============================
                #if group_idx == 0:  # Only print first group
                #    print(f"\nLooking for: res {restraint.res_i} atom '{restraint.atom_i}' <-> res {restraint.res_j} atom '{restraint.atom_j}'")
                #=====================================================================
                
                idx_i = self._get_atom_index(restraint.res_i, restraint.atom_i)
                idx_j = self._get_atom_index(restraint.res_j, restraint.atom_j)
                
                #=========================== Debug prints ============================
                #if group_idx == 0:
                #    print(f"  Found indices: {idx_i}, {idx_j}")
                #======================================================================

                #=========================== Debug prints ============================
                if group_idx == 0 and len(group_violations) < 2:  # First 2 restraints
                    print(f"      Restraint: res{restraint.res_i}:{restraint.atom_i} <-> res{restraint.res_j}:{restraint.atom_j}")
                    print(f"        Mapped to indices: {idx_i} <-> {idx_j}")
            
                #=====================================================================
                
                if idx_i is None or idx_j is None:

                    #=========================== Debug prints ============================
                    if group_idx == 0 and len(group_violations) < 2:
                        print(f"        ⚠️  Atom mapping failed!")
                    #=====================================================================
                    continue
                
                # Compute distance in Angstroms
                pos_i = coords[:, idx_i, :]
                pos_j = coords[:, idx_j, :]
                dist = torch.norm(pos_j - pos_i, dim=-1)
                dist_nm = dist / 10.0
                
                #=========================== Debug prints ============================
                if group_idx == 0 and len(group_violations) == 0:
                    print(f"        Distance: {dist.item():.2f} Å ({dist_nm.item():.3f} nm)")
                    print(f"        Target r4: {restraint.r4:.3f} nm")
                #======================================================================

                # Flat-bottom violation
                violation = torch.relu(dist_nm - restraint.r4)
                group_violations.append(violation)
            
            if not group_violations:
                continue
            
            # === MELD OR LOGIC: Use MINIMUM violation ===
            min_violation = torch.min(torch.stack(group_violations), dim=0)[0]
            energy = 0.5 * restraint.k * (min_violation ** 2)

            #===================================== Debug prints ============================
            if group_idx < 3:
                print(f"      Min violation: {min_violation.item():.4f} nm")
                print(f"      Energy contribution: {energy.item():.2f} kJ/mol")
            #===============================================================================

            total_energy = total_energy + energy
            has_contribution = True
        
        #==================================== Debug prints ============================
        print(f"    Total distance energy: {total_energy.item():.2f} kJ/mol")
        print(f"    Contributions: {has_contribution}")
        #=============================================================================

        # If no restraints contributed, create a dummy gradient connection
        if not has_contribution:
            total_energy = total_energy + 0.0 * coords[:, 0, 0]
        
        return total_energy
    
    def compute_torsion_energy(
        self,
        coords: torch.Tensor,
        active_groups: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute torsion angle restraint energy.
        
        E = k * (delta ** 2) / (2 * phi_std ** 2)
        
        Args:
            coords: [batch, n_atoms, 3] atom coordinates
            active_groups: Which restraint groups to use
        
        Returns:
            energy: [batch] total energy in kJ/mol
        """
        if active_groups is None:
            active_groups = list(range(self.n_torsion_active))
        
        batch_size = coords.shape[0]
        total_energy = torch.zeros(batch_size, device=coords.device, dtype=coords.dtype)
        has_contribution = False

        #==================================== Debug prints ============================
        if len(active_groups) > 0:
            print(f"\n  📐 Torsion Energy Computation:")
            print(f"    Active groups: {len(active_groups)}")
        #==============================================================================
        
        for group_idx in active_groups:
            if group_idx >= len(self.torsion_groups):
                continue
            
            group = self.torsion_groups[group_idx]

            #==================================== Debug prints ============================
            if group_idx < 2:  # First 2 groups
                print(f"    Group {group_idx}: {len(group)} restraints")
            #==============================================================================
            
            for restraint in group:
                idx1 = self._get_atom_index(restraint.res1, restraint.atom1)
                idx2 = self._get_atom_index(restraint.res2, restraint.atom2)
                idx3 = self._get_atom_index(restraint.res3, restraint.atom3)
                idx4 = self._get_atom_index(restraint.res4, restraint.atom4)
                
                if None in [idx1, idx2, idx3, idx4]:
                    continue
                
                # Compute dihedral angle
                phi = self._compute_dihedral(
                    coords[:, idx1, :],
                    coords[:, idx2, :],
                    coords[:, idx3, :],
                    coords[:, idx4, :]
                )
                
                # Convert to degrees
                phi_deg = torch.rad2deg(phi)

                #======================================== Debug prints ============================
                if group_idx == 0 and has_contribution == False:
                    print(f"      Angle: {phi_deg.item():.2f}°")
                    print(f"      Target: [{restraint.phi_min:.2f}, {restraint.phi_max:.2f}]°")
                #==================================================================================
                
                # Circular distance
                delta = phi_deg - restraint.phi_mean
                delta = torch.atan2(torch.sin(torch.deg2rad(delta)),
                                   torch.cos(torch.deg2rad(delta)))
                delta = torch.rad2deg(delta)
                
                # Energy
                energy = restraint.k * (delta ** 2) / (2 * restraint.phi_std ** 2)
                total_energy = total_energy + energy
                has_contribution = True
        
        #==================================== Debug prints ============================
        print(f"    Total torsion energy: {total_energy.item():.2f} kJ/mol")
        #==============================================================================

        # If no restraints contributed, create a dummy gradient connection
        if not has_contribution:
            total_energy = total_energy + 0.0 * coords[:, 0, 0]
        
        return total_energy
    
    @staticmethod
    def _compute_dihedral(p0, p1, p2, p3):
        """Compute dihedral angle between 4 points"""
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        
        b1_norm = b1 / (torch.norm(b1, dim=-1, keepdim=True) + 1e-8)
        
        v = b0 - torch.sum(b0 * b1_norm, dim=-1, keepdim=True) * b1_norm
        w = b2 - torch.sum(b2 * b1_norm, dim=-1, keepdim=True) * b1_norm
        
        x = torch.sum(v * w, dim=-1)
        y = torch.sum(torch.cross(b1_norm, v, dim=-1) * w, dim=-1)
        
        return torch.atan2(y, x)
    
    def compute_total_energy(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute total NMR restraint energy"""
        e_dist = self.compute_distance_energy(coords)
        e_tors = self.compute_torsion_energy(coords)
        return e_dist + e_tors
    
    def compute_gradient(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of NMR energy w.r.t. coordinates.
        
        Returns:
            grad: [batch, n_atoms, 3] gradient
        """
        # Create a copy with gradient tracking
        coords_copy = coords.clone().detach()
        coords_copy.requires_grad_(True)

        #========================== Debug prints ==========================
        print(f"\n  NMR Energy Computation:")
        print(f"    Coords shape: {coords.shape}")
        print(f"    Coords range: [{coords.min():.2f}, {coords.max():.2f}] Å")
        #==================================================================
        
        # Compute energy with gradient tracking
        energy = self.compute_total_energy(coords_copy)
        energy_sum = energy.sum()

        #========================== Debug prints ==========================
        print(f"    Total NMR energy: {energy_sum.item():.2f} kJ/mol")
        #==================================================================
        
        # Compute gradient
        grad = torch.autograd.grad(energy_sum, coords_copy, create_graph=False)[0]

        #========================== Debug prints ==========================
        print(f"    Gradient norm: {torch.norm(grad).item():.8f}")
        print(f"    Gradient range: [{grad.min():.8f}, {grad.max():.8f}]")
        #==================================================================
        
        return grad