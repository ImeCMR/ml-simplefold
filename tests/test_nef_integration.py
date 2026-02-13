"""
Unit tests for NEF utility functions
Tests r^-6 averaging, restraint validation, multi-assignment parsing, and more
"""

import pytest
import numpy as np
from src.simplefold.utils.nef_utils import (
    calculate_effective_distance_r6,
    validate_restraint,
    parse_restraint_assignments,
    filter_nef_restraints,
    group_restraints_by_id,
    check_restraint_consistency
)


def test_effective_distance_r6():
    """Test r^-6 averaging"""
    distances = [2.5, 3.0, 3.5]
    d_eff = calculate_effective_distance_r6(distances)
    
    # Effective distance with r^-6 averaging is LESS than the shortest distance
    # because r^-6 heavily weights shorter distances
    assert d_eff < 2.5, f"Expected d_eff < 2.5, got {d_eff}"
    assert d_eff > 2.0, f"Expected d_eff > 2.0, got {d_eff}"  # Reasonable lower bound
    print(f"✓ Effective distance: {d_eff:.2f} Å")
    
    # Test edge cases
    # Single distance
    assert abs(calculate_effective_distance_r6([3.0]) - 3.0) < 1e-5
    print("✓ Single distance works correctly")
    
    # Empty list should raise error
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_effective_distance_r6([])
    print("✓ Empty list raises ValueError")
    
    # Negative distance should raise error
    with pytest.raises(ValueError, match="must be positive"):
        calculate_effective_distance_r6([2.0, -1.0, 3.0])
    print("✓ Negative distance raises ValueError")


def test_restraint_validation():
    """Test restraint validation"""
    valid_restraint = {
        'atom1_residue_number': 5,
        'atom2_residue_number': 12,
        'atom1_atom_name': 'HD1',
        'atom2_atom_name': 'HE3',
        'upper_bound': 5.0
    }
    is_valid, msg = validate_restraint(valid_restraint, sequence_length=50)
    assert is_valid, f"Valid restraint rejected: {msg}"
    print("✓ Valid restraint accepted")
    
    # Test invalid restraint (out of range)
    invalid_restraint = valid_restraint.copy()
    invalid_restraint['atom1_residue_number'] = 100
    is_valid, msg = validate_restraint(invalid_restraint, sequence_length=50)
    assert not is_valid, "Invalid restraint should be rejected"
    print(f"✓ Invalid restraint rejected: {msg}")
    
    # Test missing upper bound
    invalid_restraint = valid_restraint.copy()
    del invalid_restraint['upper_bound']
    is_valid, msg = validate_restraint(invalid_restraint, sequence_length=50)
    assert not is_valid
    assert "Missing upper_bound" in msg
    print("✓ Missing upper_bound detected")
    
    # Test upper bound too small
    invalid_restraint = valid_restraint.copy()
    invalid_restraint['upper_bound'] = 1.0
    is_valid, msg = validate_restraint(invalid_restraint, sequence_length=50)
    assert not is_valid
    assert "too small" in msg
    print("✓ Upper bound too small detected")
    
    # Test upper bound too large
    invalid_restraint = valid_restraint.copy()
    invalid_restraint['upper_bound'] = 150.0
    is_valid, msg = validate_restraint(invalid_restraint, sequence_length=50)
    assert not is_valid
    assert "too large" in msg
    print("✓ Upper bound too large detected")
    
    # Test self-restraint
    invalid_restraint = valid_restraint.copy()
    invalid_restraint['atom1_residue_number'] = 5
    invalid_restraint['atom2_residue_number'] = 5
    invalid_restraint['atom1_atom_name'] = 'HD1'
    invalid_restraint['atom2_atom_name'] = 'HD1'
    is_valid, msg = validate_restraint(invalid_restraint, sequence_length=50)
    assert not is_valid
    assert "Self-restraint" in msg
    print("✓ Self-restraint detected")
    
    # Test with atom_set
    atom_set = {'HD1', 'HE3', 'HB1', 'HG1'}
    is_valid, msg = validate_restraint(valid_restraint, sequence_length=50, atom_set=atom_set)
    assert is_valid
    print("✓ Valid atom names accepted")
    
    invalid_restraint = valid_restraint.copy()
    invalid_restraint['atom1_atom_name'] = 'INVALID'
    is_valid, msg = validate_restraint(invalid_restraint, sequence_length=50, atom_set=atom_set)
    assert not is_valid
    assert "not found in atom set" in msg
    print("✓ Invalid atom name detected")


def test_multi_assignment_parsing():
    """Test multi-assignment restraint parsing"""
    # Test with multiple assignments
    restraint_multi = {
        'id': 'noe_1',
        'assignments': [
            {
                'atom1_residue_number': 5,
                'atom1_atom_name': 'HD1',
                'atom2_residue_number': 12,
                'atom2_atom_name': 'HE3',
                'upper_bound': 5.0
            },
            {
                'atom1_residue_number': 5,
                'atom1_atom_name': 'HD2',
                'atom2_residue_number': 12,
                'atom2_atom_name': 'HE3',
                'upper_bound': 5.0
            }
        ]
    }
    assignments = parse_restraint_assignments(restraint_multi)
    assert len(assignments) == 2, f"Expected 2 assignments, got {len(assignments)}"
    print("✓ Multi-assignment parsing works")
    
    # Test with single assignment
    restraint_single = {
        'id': 'noe_2',
        'atom1_residue_number': 5,
        'atom1_atom_name': 'HD1',
        'atom2_residue_number': 12,
        'atom2_atom_name': 'HE3',
        'upper_bound': 5.0
    }
    assignments = parse_restraint_assignments(restraint_single)
    assert len(assignments) == 1, f"Expected 1 assignment, got {len(assignments)}"
    assert assignments[0]['atom1_residue_number'] == 5
    assert assignments[0]['upper_bound'] == 5.0
    print("✓ Single assignment parsing works")


def test_filter_restraints():
    """Test restraint filtering"""
    restraints = [
        {
            'atom1_residue_number': 5,
            'atom2_residue_number': 12,
            'atom1_atom_name': 'HD1',
            'atom2_atom_name': 'HE3',
            'upper_bound': 5.0
        },
        {  # Invalid: out of range
            'atom1_residue_number': 100,
            'atom2_residue_number': 12,
            'atom1_atom_name': 'HD1',
            'atom2_atom_name': 'HE3',
            'upper_bound': 5.0
        },
        {  # Valid
            'atom1_residue_number': 7,
            'atom2_residue_number': 15,
            'atom1_atom_name': 'HB1',
            'atom2_atom_name': 'HG1',
            'upper_bound': 4.5
        }
    ]
    
    filtered, stats = filter_nef_restraints(restraints, sequence_length=50)
    
    assert stats['total'] == 3
    assert stats['valid'] == 2
    assert stats['filtered'] == 1
    assert len(filtered) == 2
    print(f"✓ Filtered {stats['filtered']}/{stats['total']} restraints")
    print(f"✓ Filter reasons: {stats['reasons']}")


def test_group_restraints_by_id():
    """Test grouping restraints by ID"""
    restraints = [
        {'id': 'noe_1', 'upper_bound': 5.0},
        {'id': 'noe_2', 'upper_bound': 4.5},
        {'id': 'noe_3', 'upper_bound': 6.0}
    ]
    
    grouped = group_restraints_by_id(restraints)
    assert len(grouped) == 3
    assert 'noe_1' in grouped
    assert grouped['noe_2']['upper_bound'] == 4.5
    print("✓ Restraint grouping works")
    
    # Test missing ID
    with pytest.raises(ValueError, match="missing 'id' field"):
        group_restraints_by_id([{'upper_bound': 5.0}])
    print("✓ Missing ID detected")
    
    # Test duplicate ID
    duplicate_restraints = [
        {'id': 'noe_1', 'upper_bound': 5.0},
        {'id': 'noe_1', 'upper_bound': 4.5}
    ]
    with pytest.raises(ValueError, match="Duplicate restraint ID"):
        group_restraints_by_id(duplicate_restraints)
    print("✓ Duplicate ID detected")


def test_check_restraint_consistency():
    """Test restraint consistency checking"""
    restraints = [
        {
            'atom1_residue_number': 5,
            'atom2_residue_number': 12,
            'upper_bound': 5.0
        },
        {
            'atom1_residue_number': 7,
            'atom2_residue_number': 8,  # Short range
            'upper_bound': 4.5
        },
        {
            'atom1_residue_number': 10,
            'atom2_residue_number': 25,  # Long range
            'upper_bound': 6.0
        },
        {
            'id': 'ambiguous',
            'assignments': [
                {'atom1_residue_number': 5, 'atom2_residue_number': 12, 'upper_bound': 5.0},
                {'atom1_residue_number': 5, 'atom2_residue_number': 13, 'upper_bound': 5.0}
            ],
            'upper_bound': 5.0
        }
    ]
    
    stats = check_restraint_consistency(restraints)
    
    assert stats['total_restraints'] == 4
    assert stats['ambiguous_restraints'] == 1
    assert 'min_distance_bound' in stats
    assert 'max_distance_bound' in stats
    assert 'mean_distance_bound' in stats
    assert stats['short_range_le5'] >= 1
    assert stats['long_range_gt10'] >= 1
    
    print(f"✓ Consistency check stats: {stats}")
    
    # Test empty restraints
    empty_stats = check_restraint_consistency([])
    assert empty_stats == {}
    print("✓ Empty restraint list handled")


if __name__ == "__main__":
    print("Running NEF integration tests...\n")
    test_effective_distance_r6()
    print()
    test_restraint_validation()
    print()
    test_multi_assignment_parsing()
    print()
    test_filter_restraints()
    print()
    test_group_restraints_by_id()
    print()
    test_check_restraint_consistency()
    print("\n✅ All unit tests passed!")