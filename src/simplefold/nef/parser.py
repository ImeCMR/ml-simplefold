#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
NEF (NMR Exchange Format) Parser

This module parses NEF files and extracts distance restraints for NOESY data.
NEF uses the STAR file format for NMR experimental data.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ResidueAtom:
    """Identifies a specific atom in a specific residue."""
    chain_code: str
    sequence_code: int  # Residue number
    atom_name: str  # e.g., "CA", "CB", "HG"
    
    def __repr__(self) -> str:
        return f"{self.chain_code}{self.sequence_code}{self.atom_name}"


@dataclass
class DistanceRestraint:
    """A single distance restraint from NOESY data."""
    atom1: ResidueAtom
    atom2: ResidueAtom
    lower_bound: float  # Angstroms
    upper_bound: float  # Angstroms
    weight: float = 1.0
    restraint_id: Optional[int] = None  # For tracking ambiguous restraints
    
    def __repr__(self) -> str:
        return f"[{self.atom1} - {self.atom2}: {self.lower_bound:.2f}-{self.upper_bound:.2f} Å]"


class NEFParser:
    """
    Parser for NMR Exchange Format (NEF) files.
    
    NEF uses the STAR file format for storing NMR data. This parser extracts
    distance restraints from nef_distance_restraint_list saveframes.
    Supports both standard NEF and CYANA-style formats.
    """
    
    def __init__(self, nef_file: Path):
        """
        Parameters
        ----------
        nef_file : Path
            Path to NEF file in STAR format
        """
        self.nef_file = Path(nef_file)
        self.restraints: List[DistanceRestraint] = []
        self._parse()
    
    def _parse(self) -> None:
        """Parse the NEF file and extract distance restraints."""
        if not self.nef_file.exists():
            raise FileNotFoundError(f"NEF file not found: {self.nef_file}")
        
        with open(self.nef_file, 'r') as f:
            content = f.read()
        
        # Extract all distance restraint list saveframes
        self._extract_distance_restraint_saveframes(content)
    
    def _extract_distance_restraint_saveframes(self, content: str) -> None:
        """Extract all nef_distance_restraint_list saveframes from the NEF file."""
        # Pattern to match ANY saveframe with its content
        # This matches: save_<framecode> <content> save_
        # Works for both standard NEF (save_nef_distance_restraint_list_*)
        # and CYANA format (save_CYANA_distance_restraints_*)
        saveframe_pattern = r'save_(\S+)\s+(.*?)\s+save_'
        all_saveframes = re.findall(saveframe_pattern, content, re.DOTALL)
        
        if not all_saveframes:
            print("Warning: No saveframes found in NEF file")
            return
        
        found_restraints = False
        for framecode, saveframe_content in all_saveframes:
            # Check if this saveframe contains distance restraints by looking for the category tag
            # This works for both standard NEF and CYANA formats
            if '_nef_distance_restraint_list.sf_category' in saveframe_content:
                self._parse_distance_restraint_saveframe(saveframe_content, framecode)
                found_restraints = True
        
        if not found_restraints:
            print("Warning: No nef_distance_restraint_list saveframes found in NEF file")
    
    def _parse_distance_restraint_saveframe(self, content: str, framecode: str) -> None:
        """Parse a single distance restraint saveframe."""
        # Find the loop containing distance restraints
        loop_pattern = r'loop_(.*?)stop_'
        loops = re.findall(loop_pattern, content, re.DOTALL)
        
        for loop_content in loops:
            # Check if this loop contains distance restraint tags
            if '_nef_distance_restraint' in loop_content:
                self._parse_distance_restraint_loop(loop_content, framecode)
    
    def _parse_distance_restraint_loop(self, loop_content: str, framecode: str) -> None:
        """Parse the distance restraint loop structure."""
        lines = loop_content.strip().split('\n')
        
        # Extract column headers (tags)
        headers = []
        data_start_idx = 0
        
        for idx, line in enumerate(lines):
            line = line.strip()
            if line.startswith('_nef_distance_restraint.'):
                # Extract the tag name after the dot
                tag = line.split('.')[1]
                headers.append(tag)
            elif line and not line.startswith('_'):
                # First data line
                data_start_idx = idx
                break
        
        if not headers:
            print(f"Warning: No headers found in distance restraint loop in {framecode}")
            return
        
        # Create a mapping of tag names to column indices
        col_map = {tag: idx for idx, tag in enumerate(headers)}
        
        # Required columns
        required = ['chain_code_1', 'sequence_code_1', 'atom_name_1',
                   'chain_code_2', 'sequence_code_2', 'atom_name_2']
        
        if not all(col in col_map for col in required):
            print(f"Warning: Missing required columns in {framecode}")
            return
        
        # Parse data rows
        for line in lines[data_start_idx:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                restraint = self._parse_restraint_data_line(line, col_map)
                if restraint is not None:
                    self.restraints.append(restraint)
            except Exception as e:
                print(f"Warning: Failed to parse restraint line: {e}")
                print(f"  Line: {line}")
    
    def _parse_restraint_data_line(self, line: str, col_map: Dict[str, int]) -> Optional[DistanceRestraint]:
        """Parse a single data line from the distance restraint loop."""
        # Split the line into values, handling quoted strings and dots
        values = self._split_data_line(line)
        
        if len(values) < len(col_map):
            return None
        
        try:
            # Extract atom 1
            chain_code_1 = self._get_value(values, col_map, 'chain_code_1', 'A')
            sequence_code_1 = int(self._get_value(values, col_map, 'sequence_code_1', '1'))
            atom_name_1 = self._get_value(values, col_map, 'atom_name_1', 'CA')
            
            # Extract atom 2
            chain_code_2 = self._get_value(values, col_map, 'chain_code_2', 'A')
            sequence_code_2 = int(self._get_value(values, col_map, 'sequence_code_2', '1'))
            atom_name_2 = self._get_value(values, col_map, 'atom_name_2', 'CA')
            
            # Extract bounds - handle different column name conventions
            # Some NEF files use 'lower_limit' and 'upper_limit'
            # Others use 'lower_linear_limit' and 'upper_linear_limit'
            # CYANA format may use 'target_value' with uncertainty
            
            lower_limit = None
            upper_limit = None
            
            # Try different column names for lower bound
            for col_name in ['lower_limit', 'lower_linear_limit', 'lower_bound']:
                lower_limit = self._get_value(values, col_map, col_name, None)
                if lower_limit not in [None, '.', '']:
                    break
            
            # Try different column names for upper bound
            for col_name in ['upper_limit', 'upper_linear_limit', 'upper_bound']:
                upper_limit = self._get_value(values, col_map, col_name, None)
                if upper_limit not in [None, '.', '']:
                    break
            
            # Handle target_value format (CYANA style)
            if (lower_limit is None or lower_limit in ['.', '']) and \
               (upper_limit is None or upper_limit in ['.', '']):
                target_value = self._get_value(values, col_map, 'target_value', None)
                target_uncertainty = self._get_value(values, col_map, 'target_value_uncertainty', None)
                
                if target_value and target_value not in ['.', '']:
                    target = float(target_value)
                    uncertainty = float(target_uncertainty) if target_uncertainty and target_uncertainty != '.' else 0.5
                    lower_limit = max(1.8, target - uncertainty)  # Don't go below 1.8 Å
                    upper_limit = target + uncertainty
            
            # Handle missing bounds with defaults
            if lower_limit is None or lower_limit in ['.', '']:
                lower_bound = 1.8
            else:
                lower_bound = float(lower_limit)
            
            if upper_limit is None or upper_limit in ['.', '']:
                upper_bound = 6.0
            else:
                upper_bound = float(upper_limit)
            
            # Extract weight
            weight_val = self._get_value(values, col_map, 'weight', '1.0')
            weight = 1.0 if weight_val == '.' else float(weight_val)
            
            # Extract restraint_id for tracking ambiguous restraints
            restraint_id_val = self._get_value(values, col_map, 'restraint_id', None)
            restraint_id = None if restraint_id_val is None or restraint_id_val == '.' else int(restraint_id_val)
            
            # Create restraint
            atom1 = ResidueAtom(chain_code_1, sequence_code_1, atom_name_1)
            atom2 = ResidueAtom(chain_code_2, sequence_code_2, atom_name_2)
            
            return DistanceRestraint(
                atom1=atom1,
                atom2=atom2,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                weight=weight,
                restraint_id=restraint_id
            )
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing restraint values: {e}")
            return None
    
    def _split_data_line(self, line: str) -> List[str]:
        """
        Split a data line into values, handling quoted strings.
        In STAR format, values can be:
        - Unquoted (no spaces)
        - Single quoted ('value')
        - Double quoted ("value")
        - Dot (.) for null/missing
        """
        values = []
        current = []
        in_quotes = None
        
        i = 0
        while i < len(line):
            char = line[i]
            
            if in_quotes:
                if char == in_quotes:
                    in_quotes = None
                else:
                    current.append(char)
            elif char in ('"', "'"):
                in_quotes = char
            elif char.isspace():
                if current:
                    values.append(''.join(current))
                    current = []
            else:
                current.append(char)
            
            i += 1
        
        if current:
            values.append(''.join(current))
        
        return values
    
    def _get_value(self, values: List[str], col_map: Dict[str, int], 
                   column: str, default: Optional[str]) -> Optional[str]:
        """Get a value from the data row, with fallback to default."""
        if column not in col_map:
            return default
        
        idx = col_map[column]
        if idx >= len(values):
            return default
        
        value = values[idx]
        if value == '.':
            return default
        
        return value
    
    def get_restraints(self) -> List[DistanceRestraint]:
        """Get parsed distance restraints."""
        return self.restraints
    
    def get_ambiguous_restraints(self) -> Dict[int, List[DistanceRestraint]]:
        """
        Group restraints by restraint_id to identify ambiguous restraints.
        
        Returns
        -------
        Dict mapping restraint_id to list of alternative restraints
        """
        ambiguous = {}
        for restraint in self.restraints:
            if restraint.restraint_id is not None:
                if restraint.restraint_id not in ambiguous:
                    ambiguous[restraint.restraint_id] = []
                ambiguous[restraint.restraint_id].append(restraint)
        
        # Filter to only those with multiple alternatives
        return {rid: rlist for rid, rlist in ambiguous.items() if len(rlist) > 1}
    
    def summary(self) -> str:
        """Return a summary of parsed restraints."""
        summary = f"NEF File: {self.nef_file}\n"
        summary += f"Number of restraints: {len(self.restraints)}\n"
        
        if self.restraints:
            upper_bounds = [r.upper_bound for r in self.restraints]
            lower_bounds = [r.lower_bound for r in self.restraints]
            summary += f"Upper bound range: {min(upper_bounds):.2f} - {max(upper_bounds):.2f} Å\n"
            summary += f"Lower bound range: {min(lower_bounds):.2f} - {max(lower_bounds):.2f} Å\n"
            
            # Add ambiguous restraint info
            ambiguous = self.get_ambiguous_restraints()
            if ambiguous:
                summary += f"Number of ambiguous restraint groups: {len(ambiguous)}\n"
        
        return summary