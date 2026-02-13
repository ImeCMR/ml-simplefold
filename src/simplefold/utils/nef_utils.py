#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
Utility functions for NEF (NMR Exchange Format) file processing.
Provides helper functions for restraint processing, validation, and conversion.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


def calculate_effective_distance_r6(distances: List[float]) -> float:
    """
    Calculate the r^-6 summed effective distance from a list of distances.
    
    This implements the standard NMR ensemble averaging using the r^-6 sum,
    which accounts for the distance dependence of NOE intensities.
    
    Parameters
    ----------
    distances : List[float]
        List of distances (in Angstroms) for ambiguous restraint assignments.
        
    Returns
    -------
    float
        The effective distance computed as: (sum(d_i^-6))^(-1/6)
        
    Raises
    ------
    ValueError
        If distances list is empty or contains invalid values.
        
    Examples
    --------
    >>> distances = [2.5, 3.0, 3.5]
    >>> d_eff = calculate_effective_distance_r6(distances)
    """
    if not distances:
        raise ValueError("distances list cannot be empty")
    
    distances = np.asarray(distances, dtype=np.float32)
    
    if np.any(distances <= 0):
        raise ValueError("All distances must be positive")
    
    # Compute r^-6 sum
    distances_minus_6 = distances ** (-6)
    sum_distances_minus_6 = np.sum(distances_minus_6)
    
    # Compute effective distance as (sum(r^-6))^(-1/6)
    effective_distance = sum_distances_minus_6 ** (-1/6)
    
    return float(effective_distance)


def validate_restraint(
    restraint: Dict,
    sequence_length: int,
    atom_set: Optional[set] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a single distance restraint against various criteria.
    
    Checks:
    - Residue indices are within valid range
    - Upper bound is physically reasonable (>1.5 Å)
    - Not a self-restraint (same atom)
    - Atom names are recognized (if atom_set provided)
    
    Parameters
    ----------
    restraint : Dict
        Restraint dictionary with keys: atom1_residue_number, atom2_residue_number,
        atom1_atom_name, atom2_atom_name, upper_bound
    sequence_length : int
        Length of the protein sequence (for residue validation)
    atom_set : Optional[set]
        Set of valid atom names. If None, skips atom name validation.
        
    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_valid, error_message). If valid, error_message is None.
        
    Examples
    --------
    >>> restraint = {
    ...     'atom1_residue_number': 5,
    ...     'atom2_residue_number': 12,
    ...     'atom1_atom_name': 'HD1',
    ...     'atom2_atom_name': 'HE3',
    ...     'upper_bound': 5.0
    ... }
    >>> is_valid, msg = validate_restraint(restraint, 50)
    >>> assert is_valid
    """
    res1 = restraint.get('atom1_residue_number')
    res2 = restraint.get('atom2_residue_number')
    atom1 = restraint.get('atom1_atom_name')
    atom2 = restraint.get('atom2_atom_name')
    ub = restraint.get('upper_bound')
    
    # Check residue numbers
    if res1 is None or res2 is None:
        return False, "Missing residue numbers"
    
    if res1 < 1 or res1 > sequence_length:
        return False, f"Residue 1 {res1} out of range [1, {sequence_length}]"
    
    if res2 < 1 or res2 > sequence_length:
        return False, f"Residue 2 {res2} out of range [1, {sequence_length}]"
    
    # Check upper bound
    if ub is None:
        return False, "Missing upper_bound"
    
    if ub < 1.5:
        return False, f"Upper bound {ub} Å is too small (< 1.5 Å)"
    
    if ub > 100.0:
        return False, f"Upper bound {ub} Å is too large (> 100 Å)"
    
    # Check for self-restraints
    if res1 == res2 and atom1 == atom2:
        return False, "Self-restraint (same atom to itself)"
    
    # Check atom names if provided
    if atom_set is not None:
        if atom1 not in atom_set:
            return False, f"Atom '{atom1}' not found in atom set"
        if atom2 not in atom_set:
            return False, f"Atom '{atom2}' not found in atom set"
    
    return True, None


def filter_nef_restraints(
    restraints: List[Dict],
    sequence_length: int,
    atom_set: Optional[set] = None,
    verbose: bool = False,
) -> Tuple[List[Dict], Dict]:
    """
    Filter out invalid restraints from a NEF restraint list.
    
    Parameters
    ----------
    restraints : List[Dict]
        List of restraint dictionaries
    sequence_length : int
        Length of the protein sequence
    atom_set : Optional[set]
        Set of valid atom names. If None, skips atom name validation.
    verbose : bool
        If True, print warnings for filtered restraints
        
    Returns
    -------
    Tuple[List[Dict], Dict]
        (filtered_restraints, filter_stats) where filter_stats contains:
        - 'total': total input restraints
        - 'valid': number of valid restraints
        - 'filtered': number of filtered restraints
        - 'reasons': dict of filter reasons and counts
        
    Examples
    --------
    >>> restraints = [...]  # List of restraints
    >>> filtered, stats = filter_nef_restraints(restraints, 50)
    >>> print(f"Kept {stats['valid']} / {stats['total']} restraints")
    """
    valid_restraints = []
    filter_reasons = {}
    
    for restraint in restraints:
        is_valid, error_msg = validate_restraint(restraint, sequence_length, atom_set)
        
        if is_valid:
            valid_restraints.append(restraint)
        else:
            filter_reasons[error_msg] = filter_reasons.get(error_msg, 0) + 1
            if verbose:
                rid = restraint.get('id', 'unknown')
                warnings.warn(f"Filtering restraint {rid}: {error_msg}")
    
    stats = {
        'total': len(restraints),
        'valid': len(valid_restraints),
        'filtered': len(restraints) - len(valid_restraints),
        'reasons': filter_reasons,
    }
    
    return valid_restraints, stats


def parse_restraint_assignments(
    restraint: Dict,
) -> List[Dict]:
    """
    Extract all possible atom pair assignments from a restraint.
    
    Handles both simple restraints (single assignment) and ambiguous restraints
    (multiple possible assignments).
    
    Parameters
    ----------
    restraint : Dict
        A restraint dictionary. Can have either:
        - Simple format: atom1_residue_number, atom1_atom_name, etc.
        - Ambiguous format: 'assignments' key with list of assignments
        
    Returns
    -------
    List[Dict]
        List of assignment dictionaries, each with keys:
        atom1_residue_number, atom1_atom_name,
        atom2_residue_number, atom2_atom_name, upper_bound
        
    Examples
    --------
    >>> restraint = {
    ...     'id': 'noe_1',
    ...     'assignments': [
    ...         {'atom1_residue_number': 5, 'atom1_atom_name': 'HD1', ...},
    ...         {'atom1_residue_number': 5, 'atom1_atom_name': 'HD2', ...},
    ...     ]
    ... }
    >>> assignments = parse_restraint_assignments(restraint)
    >>> assert len(assignments) == 2
    """
    # Check if restraint has multiple assignments (ambiguous)
    if 'assignments' in restraint:
        return restraint['assignments']
    
    # Otherwise, treat as single assignment
    assignment = {
        'atom1_residue_number': restraint.get('atom1_residue_number'),
        'atom1_atom_name': restraint.get('atom1_atom_name'),
        'atom2_residue_number': restraint.get('atom2_residue_number'),
        'atom2_atom_name': restraint.get('atom2_atom_name'),
        'upper_bound': restraint.get('upper_bound'),
    }
    return [assignment]


def group_restraints_by_id(
    restraints: List[Dict],
) -> Dict[str, Dict]:
    """
    Group restraints by their ID for easier lookup.
    
    Parameters
    ----------
    restraints : List[Dict]
        List of restraint dictionaries
        
    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping restraint ID to restraint
        
    Raises
    ------
    ValueError
        If duplicate restraint IDs are found
    """
    grouped = {}
    
    for restraint in restraints:
        rid = restraint.get('id')
        if rid is None:
            raise ValueError("Restraint missing 'id' field")
        
        if rid in grouped:
            raise ValueError(f"Duplicate restraint ID: {rid}")
        
        grouped[rid] = restraint
    
    return grouped


def check_restraint_consistency(
    restraints: List[Dict],
) -> Dict:
    """
    Check consistency of restraint collection.
    
    Reports statistics and potential issues:
    - Distance range (min, max, mean upper bounds)
    - Residue span (intra-residual, short-range, long-range)
    - Ambiguous restraints count
    
    Parameters
    ----------
    restraints : List[Dict]
        List of restraint dictionaries
        
    Returns
    -------
    Dict
        Statistics dictionary with keys:
        - 'total_restraints': total count
        - 'min_distance_bound': minimum upper bound
        - 'max_distance_bound': maximum upper bound
        - 'mean_distance_bound': mean upper bound
        - 'intra_residual': count of same-residue restraints
        - 'short_range': count with |i-j| <= 5
        - 'medium_range': count with 5 < |i-j| <= 10
        - 'long_range': count with |i-j| > 10
        - 'ambiguous_restraints': count with multiple assignments
    """
    if not restraints:
        return {}
    
    distance_bounds = []
    residue_spans = []
    ambiguous_count = 0
    intra = 0
    short = 0
    medium = 0
    long = 0
    
    for restraint in restraints:
        # Collect distance bounds
        ub = restraint.get('upper_bound')
        if ub is not None:
            distance_bounds.append(ub)
        
        # Check ambiguity
        if 'assignments' in restraint and len(restraint['assignments']) > 1:
            ambiguous_count += 1
        
        # Check residue span
        res1 = restraint.get('atom1_residue_number')
        res2 = restraint.get('atom2_residue_number')
        
        if res1 is not None and res2 is not None:
            span = abs(res1 - res2)
            residue_spans.append(span)
            
            if span == 0:
                intra += 1
            elif span <= 5:
                short += 1
            elif span <= 10:
                medium += 1
            else:
                long += 1
    
    stats = {
        'total_restraints': len(restraints),
        'ambiguous_restraints': ambiguous_count,
        'intra_residual': intra,
        'short_range_le5': short,
        'medium_range_5_10': medium,
        'long_range_gt10': long,
    }
    
    if distance_bounds:
        stats.update({
            'min_distance_bound': float(np.min(distance_bounds)),
            'max_distance_bound': float(np.max(distance_bounds)),
            'mean_distance_bound': float(np.mean(distance_bounds)),
            'median_distance_bound': float(np.median(distance_bounds)),
        })
    
    if residue_spans:
        stats.update({
            'min_residue_span': int(np.min(residue_spans)),
            'max_residue_span': int(np.max(residue_spans)),
            'mean_residue_span': float(np.mean(residue_spans)),
        })
    
    return stats
