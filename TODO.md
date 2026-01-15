# SimpleNOEFold - Next Steps After Applying Patches

**Status:** ✅ All patches from CODE_PATCHES.md applied  
**Date:** January 2026  
**Goal:** Validate implementation and prepare for production

---

## Table of Contents

1. [Immediate Next Steps (Today)](#immediate-next-steps-today)
2. [Short-Term Next Steps (This Week)](#short-term-next-steps-this-week)
3. [Medium-Term Next Steps (Next 1-2 Weeks)](#medium-term-next-steps-next-1-2-weeks)
4. [Testing Checklist](#testing-checklist)
5. [Common Issues & Solutions](#common-issues--solutions)
6. [Success Criteria](#success-criteria)
7. [Quick Debug Commands](#quick-debug-commands)
8. [Timeline Summary](#timeline-summary)

---

## Immediate Next Steps (Today)

### Step 1: Verify All Patches Applied Correctly (15 min)

Run a quick syntax check to ensure no Python errors:

```bash
# Check for Python syntax errors in patched files
python -m py_compile src/simplefold/model/torch/bayesian_steering.py
python -m py_compile src/simplefold/model/torch/sampler.py
python -m py_compile src/simplefold/model/simplefold.py
python -m py_compile src/simplefold/utils/datamodule_utils.py
python -m py_compile src/simplefold/utils/nef_utils.py
```

**Expected output:** No errors

**If errors appear:** Review the patch application and fix syntax issues before proceeding.

---

### Step 2: Create Unit Tests (1-2 hours)

Create `tests/test_nef_integration.py`:

```python
"""
Unit tests for NEF utility functions
Tests r^-6 averaging, restraint validation, and multi-assignment parsing
"""

import torch
import pytest
from src.simplefold.utils.nef_utils import (
    calculate_effective_distance_r6,
    validate_restraint,
    parse_restraint_assignments,
    filter_nef_restraints
)

def test_effective_distance_r6():
    """Test r^-6 averaging"""
    distances = [2.5, 3.0, 3.5]
    d_eff = calculate_effective_distance_r6(distances)
    
    # Effective distance should favor shorter distances
    assert d_eff < 3.0, f"Expected d_eff < 3.0, got {d_eff}"
    assert d_eff > 2.5, f"Expected d_eff > 2.5, got {d_eff}"
    print(f"✓ Effective distance: {d_eff:.2f} Å")

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
        }
    ]
    
    filtered, stats = filter_nef_restraints(restraints, sequence_length=50)
    
    assert stats['total'] == 2
    assert stats['valid'] == 1
    assert stats['filtered'] == 1
    print(f"✓ Filtered {stats['filtered']}/{stats['total']} restraints")

if __name__ == "__main__":
    print("Running NEF integration tests...\n")
    test_effective_distance_r6()
    test_restraint_validation()
    test_multi_assignment_parsing()
    test_filter_restraints()
    print("\n✅ All unit tests passed!")
```

**Run the tests:**
```bash
python tests/test_nef_integration.py
```

**Expected output:**
```
Running NEF integration tests...

✓ Effective distance: 2.76 Å
✓ Valid restraint accepted
✓ Invalid restraint rejected: Residue 1 100 out of range [1, 50]
✓ Multi-assignment parsing works
✓ Single assignment parsing works
✓ Filtered 1/2 restraints

✅ All unit tests passed!
```

---

### Step 3: Test NEF File Parsing (30 min)

**Create a minimal test NEF file** at `data/test_nef/test.nef`:

```
data_nef_nmr_restraint_list

save_distance_restraints_1
  _nef_distance_restraint_list.sf_category          nef_distance_restraint_list
  _nef_distance_restraint_list.sf_framecode         distance_restraints_1

  loop_
    _nef_distance_restraint.index
    _nef_distance_restraint.restraint_id
    _nef_distance_restraint.restraint_combination_id
    _nef_distance_restraint.chain_code_1
    _nef_distance_restraint.sequence_code_1
    _nef_distance_restraint.residue_name_1
    _nef_distance_restraint.atom_name_1
    _nef_distance_restraint.chain_code_2
    _nef_distance_restraint.sequence_code_2
    _nef_distance_restraint.residue_name_2
    _nef_distance_restraint.atom_name_2
    _nef_distance_restraint.weight
    _nef_distance_restraint.target_value
    _nef_distance_restraint.target_value_uncertainty
    _nef_distance_restraint.lower_linear_limit
    _nef_distance_restraint.lower_limit
    _nef_distance_restraint.upper_limit
    _nef_distance_restraint.upper_linear_limit

    1  1  .  A  5   TYR  HD1  A  12  LEU  HE3  1.0  .  .  .  1.8  5.0  .
    2  2  .  A  7   ALA  HB1  A  15  VAL  HG1  1.0  .  .  .  1.8  4.5  .
    3  3  .  A  10  MET  HE1  A  20  PHE  HD1  1.0  .  .  .  1.8  6.0  .

  stop_

save_
```

**Create test script** `tests/test_parse_nef.py`:

```python
"""
Test NEF file parsing
"""

from src.simplefold.datasets.nef_parser import NEFParser
import os

def test_nef_parsing():
    nef_file = "data/test_nef/test.nef"
    
    # Check file exists
    if not os.path.exists(nef_file):
        print(f"❌ NEF file not found: {nef_file}")
        print("Please create the test NEF file first")
        return False
    
    # Parse NEF file
    parser = NEFParser(nef_file)
    restraints = parser.get_distance_restraints()
    
    print(f"\n✓ Loaded {len(restraints)} restraints from {nef_file}")
    
    # Display restraints
    for r in restraints:
        print(f"  Restraint {r['id']}: "
              f"{r['atom1_atom_name']}@{r['atom1_residue_number']} - "
              f"{r['atom2_atom_name']}@{r['atom2_residue_number']}, "
              f"UB={r['upper_bound']:.1f} Å")
    
    # Validate
    assert len(restraints) == 3, f"Expected 3 restraints, got {len(restraints)}"
    assert all('upper_bound' in r for r in restraints), "Missing upper_bound"
    assert all('id' in r for r in restraints), "Missing restraint IDs"
    
    print("\n✅ NEF parsing works!")
    return True

if __name__ == "__main__":
    test_nef_parsing()
```

**Run the test:**
```bash
python tests/test_parse_nef.py
```

---

### Step 4: Test Bayesian Steering Module (30 min)

Create `tests/test_bayesian_steering.py`:

```python
"""
Test Bayesian steering module
Tests forward pass, gradient computation, and energy calculation
"""

import torch
from src.simplefold.model.torch.bayesian_steering import BayesianSteering

def test_bayesian_steering():
    print("\nTesting Bayesian Steering Module...")
    
    # Create dummy data
    batch_size = 2
    num_atoms = 50
    coords = torch.randn(batch_size, num_atoms, 3) * 10.0
    coords.requires_grad = True
    
    # Create dummy restraints
    restraints = [
        [
            {
                'id': 'noe_1',
                'atom1_residue_number': 5,
                'atom1_atom_name': 'HD1',
                'atom2_residue_number': 12,
                'atom2_atom_name': 'HE3',
                'upper_bound': 5.0
            },
            {
                'id': 'noe_2',
                'atom1_residue_number': 7,
                'atom1_atom_name': 'HB1',
                'atom2_residue_number': 15,
                'atom2_atom_name': 'HG1',
                'upper_bound': 4.5
            }
        ],
        []  # Empty for second batch item
    ]
    
    # Create atom mappings (simplified for testing)
    atom_to_idx = [
        {'HD1': 5, 'HE3': 12, 'HB1': 7, 'HG1': 15},
        {}
    ]
    
    bonds = [[], []]
    atom_names = [['HD1', 'HE3', 'HB1', 'HG1'], []]
    
    # Initialize steering
    steering = BayesianSteering(
        enoe_weight=1.0,
        egeom_weight=0.0,  # Disable geometry for this test
        time_dependent_variance=True
    )
    
    # Test forward pass at different timesteps
    for t_val in [0.1, 0.5, 0.9]:
        t = torch.tensor(t_val)
        f_steering, e_bayesian = steering(
            coords, restraints, t, bonds, atom_to_idx, atom_names
        )
        
        print(f"\n  t={t_val}:")
        print(f"    ✓ Steering force shape: {f_steering.shape}")
        print(f"    ✓ Bayesian energy: {e_bayesian.item():.4f}")
        print(f"    ✓ Force magnitude: {f_steering.norm().item():.4f}")
        
        # Validate
        assert f_steering.shape == coords.shape, "Shape mismatch"
        assert not torch.isnan(e_bayesian), "NaN energy"
        assert not torch.isnan(f_steering).any(), "NaN in steering force"
        assert f_steering.requires_grad or e_bayesian.requires_grad, "No gradients"
    
    print("\n✅ Bayesian steering works correctly!")
    return True

def test_time_dependent_variance():
    """Test that variance decreases with time"""
    print("\nTesting time-dependent variance...")
    
    steering = BayesianSteering(time_dependent_variance=True)
    
    t_early = torch.tensor(0.1)
    t_late = torch.tensor(0.9)
    
    var_early = steering.get_variance(t_early)
    var_late = steering.get_variance(t_late)
    
    print(f"  Variance at t=0.1: {var_early:.4f}")
    print(f"  Variance at t=0.9: {var_late:.4f}")
    
    assert var_early > var_late, "Variance should decrease with time"
    print("  ✓ Variance decreases as expected")
    
    print("\n✅ Time-dependent variance works!")

if __name__ == "__main__":
    test_bayesian_steering()
    test_time_dependent_variance()
```

**Run the test:**
```bash
python tests/test_bayesian_steering.py
```

---

## Short-Term Next Steps (This Week)

### Step 5: Integration Test with Real Data (2-3 hours)

Test the full pipeline with minimal training:

```bash
# Create test directory structure
mkdir -p logs/test_integration

# Minimal training test (5 steps, 1 batch)
python train.py \
  --config-name=base_train \
  +data.nef_dir=./data/test_nef \
  trainer.max_steps=5 \
  trainer.limit_train_batches=1 \
  trainer.limit_val_batches=0 \
  model.bayesian_loss_weight=0.1 \
  paths.output_dir=logs/test_integration
```

**What to check in the output:**

✅ **Training starts without errors**
```
Epoch 0:   0%|          | 0/1 [00:00<?, ?it/s]
```

✅ **Bayesian loss appears in logs**
```
loss/bayesian: 0.xxxx
loss/train: x.xxxx
```

✅ **No NaN or Inf values**
```
# Check logs for NaN warnings
grep -i "nan\|inf" logs/test_integration/train.log
```

✅ **Steering force is computed**
```
# Should see steering force calculation in debug output
```

**If training completes successfully:**
```
✅ Integration test passed! Pipeline is working.
```

---

### Step 6: Validate Loss Normalization (1 hour)

Create `tests/test_loss_scaling.py`:

```python
"""
Verify that Bayesian loss scales correctly with restraint count
Loss per restraint should remain approximately constant
"""

import torch
from src.simplefold.model.torch.bayesian_steering import BayesianSteering

def test_loss_normalization():
    print("\nTesting loss normalization...")
    
    # Setup
    batch_size = 1
    num_atoms = 50
    coords = torch.randn(batch_size, num_atoms, 3) * 10.0
    coords.requires_grad = True
    
    steering = BayesianSteering(enoe_weight=1.0, egeom_weight=0.0)
    t = torch.tensor(0.5)
    
    atom_to_idx = [{'HD1': 5, 'HE3': 12, 'HB1': 7, 'HG1': 15}]
    bonds = [[]]
    atom_names = [['HD1', 'HE3', 'HB1', 'HG1']]
    
    # Test with different numbers of restraints
    results = []
    
    for num_restraints in [1, 5, 10, 20]:
        restraints = [[
            {
                'id': f'noe_{i}',
                'atom1_residue_number': 5,
                'atom1_atom_name': 'HD1',
                'atom2_residue_number': 12,
                'atom2_atom_name': 'HE3',
                'upper_bound': 5.0
            }
            for i in range(num_restraints)
        ]]
        
        _, e_bayesian = steering(coords, restraints, t, bonds, atom_to_idx, atom_names)
        
        loss_per_restraint = e_bayesian.item() / num_restraints
        results.append((num_restraints, e_bayesian.item(), loss_per_restraint))
        
        print(f"  {num_restraints:2d} restraints: "
              f"total_loss={e_bayesian.item():.4f}, "
              f"per_restraint={loss_per_restraint:.4f}")
    
    # Check that loss per restraint is approximately constant
    losses_per_restraint = [r[2] for r in results]
    mean_loss = sum(losses_per_restraint) / len(losses_per_restraint)
    std_loss = (sum((x - mean_loss)**2 for x in losses_per_restraint) / len(losses_per_restraint))**0.5
    
    print(f"\n  Mean loss per restraint: {mean_loss:.4f}")
    print(f"  Std deviation: {std_loss:.4f}")
    
    # Loss per restraint should not vary too much
    if std_loss / mean_loss < 0.1:  # Less than 10% variation
        print("  ✓ Loss normalization is good")
    else:
        print(f"  ⚠️  Warning: Loss per restraint varies significantly")
    
    print("\n✅ Loss normalization test complete!")

if __name__ == "__main__":
    test_loss_normalization()
```

**Run the test:**
```bash
python tests/test_loss_scaling.py
```

---

### Step 7: Test Steering Schedule Types (1 hour)

Test different steering schedule types to ensure they work correctly:

```bash
# Test 1: Linear schedule (default)
python train.py \
  --config-name=base_train \
  +data.nef_dir=./data/test_nef \
  trainer.max_steps=5 \
  trainer.limit_train_batches=1 \
  model.sampler.steering_schedule_type='linear' \
  paths.output_dir=logs/test_linear

# Test 2: Sigmoid schedule
python train.py \
  --config-name=base_train \
  +data.nef_dir=./data/test_nef \
  trainer.max_steps=5 \
  trainer.limit_train_batches=1 \
  model.sampler.steering_schedule_type='sigmoid' \
  paths.output_dir=logs/test_sigmoid

# Test 3: Exponential schedule
python train.py \
  --config-name=base_train \
  +data.nef_dir=./data/test_nef \
  trainer.max_steps=5 \
  trainer.limit_train_batches=1 \
  model.sampler.steering_schedule_type='exponential' \
  paths.output_dir=logs/test_exponential
```

**Create visualization script** `tests/visualize_schedules.py`:

```python
"""
Visualize different steering schedules
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_schedules():
    t_values = torch.linspace(0, 1, 100)
    gamma = 1.0
    
    schedules = {
        'linear': t_values * gamma,
        'sigmoid': torch.sigmoid(5 * (t_values - 0.5)) * gamma,
        'exponential': (torch.exp(t_values) - 1) / (np.e - 1) * gamma,
        'constant': torch.ones_like(t_values) * gamma,
        'square': (t_values ** 2) * gamma
    }
    
    plt.figure(figsize=(10, 6))
    for name, schedule in schedules.items():
        plt.plot(t_values.numpy(), schedule.numpy(), label=name, linewidth=2)
    
    plt.xlabel('Timestep (t)', fontsize=12)
    plt.ylabel('Steering Strength γ(t)', fontsize=12)
    plt.title('Steering Schedule Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('steering_schedules.png', dpi=150)
    print("✓ Saved visualization to steering_schedules.png")

if __name__ == "__main__":
    visualize_schedules()
```

**Run visualization:**
```bash
python tests/visualize_schedules.py
```

---

## Medium-Term Next Steps (Next 1-2 Weeks)

### Step 8: Full Fine-Tuning Test

Once all short-term tests pass, run a full fine-tuning experiment:

```bash
# Full fine-tuning with real NEF data
python train.py \
  --config-name=base_train \
  +data.nef_dir=./data/nef_files \
  model.bayesian_loss_weight=0.1 \
  model.sampler.steering_schedule_type='sigmoid' \
  model.sampler.steering_schedule_gamma=1.0 \
  trainer.max_steps=10000 \
  trainer.val_check_interval=500 \
  trainer.check_val_every_n_epoch=null \
  logger=tensorboard \
  paths.output_dir=logs/full_finetuning
```

**Monitor during training:**

```bash
# Watch logs
tail -f logs/full_finetuning/train.log

# Monitor with TensorBoard
tensorboard --logdir=logs/full_finetuning/tensorboard
```

**What to monitor:**

1. **Training loss** - Should decrease steadily
2. **Bayesian loss** - Should decrease over time
3. **Loss components** - Check balance between main loss and Bayesian loss
4. **Validation metrics** - Should improve
5. **GPU memory** - Should remain stable
6. **Training speed** - Note iterations/second

**Expected timeline:**
- 10,000 steps at ~1-2 it/s = 1.5-3 hours on a single GPU
- With multi-GPU: faster depending on GPU count

---

### Step 9: Inference Test with Steering

After fine-tuning completes, test inference:

```bash
# Inference with steering
python inference.py \
  --checkpoint=logs/full_finetuning/checkpoints/best.ckpt \
  --input=test_data/structure.cif \
  --nef=test_data/restraints.nef \
  --output=predictions/predicted.pdb \
  --use_steering=true
```

**Validation steps:**

1. **Check output exists**
```bash
ls -lh predictions/predicted.pdb
```

2. **Validate structure format**
```bash
# Use a structure validation tool
# e.g., pymol, biopython, etc.
```

3. **Calculate restraint violations**

Create `tests/validate_restraints.py`:

```python
"""
Calculate restraint violations in predicted structure
"""

from Bio.PDB import PDBParser
import numpy as np

def calculate_violations(pdb_file, nef_file):
    """Calculate how many restraints are violated"""
    
    # Parse structure
    parser = PDBParser()
    structure = parser.get_structure('pred', pdb_file)
    
    # Parse restraints (use your NEFParser)
    from src.simplefold.datasets.nef_parser import NEFParser
    nef_parser = NEFParser(nef_file)
    restraints = nef_parser.get_distance_restraints()
    
    violations = []
    
    for r in restraints:
        # Get atoms (simplified - you'll need proper atom lookup)
        res1 = r['atom1_residue_number']
        res2 = r['atom2_residue_number']
        atom1_name = r['atom1_atom_name']
        atom2_name = r['atom2_atom_name']
        upper_bound = r['upper_bound']
        
        # Calculate distance (you'll need to implement this properly)
        # distance = get_atom_distance(structure, res1, atom1_name, res2, atom2_name)
        
        # Check violation
        # if distance > upper_bound:
        #     violations.append({
        #         'restraint': r['id'],
        #         'distance': distance,
        #         'upper_bound': upper_bound,
        #         'violation': distance - upper_bound
        #     })
    
    return violations

# Run validation
# violations = calculate_violations('predictions/predicted.pdb', 'test_data/restraints.nef')
# print(f"Violations: {len(violations)}/{len(restraints)}")
```

---

### Step 10: Compare With/Without Steering

Compare baseline model vs. steered model:

```bash
# Baseline (no steering)
python inference.py \
  --checkpoint=logs/base_model/checkpoints/last.ckpt \
  --input=test_data/structure.cif \
  --output=predictions/baseline.pdb \
  --use_steering=false

# With steering
python inference.py \
  --checkpoint=logs/full_finetuning/checkpoints/best.ckpt \
  --input=test_data/structure.cif \
  --nef=test_data/restraints.nef \
  --output=predictions/steered.pdb \
  --use_steering=true

# Compare metrics
python scripts/compare_predictions.py \
  --baseline=predictions/baseline.pdb \
  --steered=predictions/steered.pdb \
  --reference=test_data/experimental.pdb \
  --restraints=test_data/restraints.nef
```

**Metrics to compare:**
- RMSD to experimental structure
- TM-score
- Number of restraint violations
- Magnitude of violations
- Physical plausibility (Ramachandran plot, clash score)

---

## Testing Checklist

Use this checklist to track your progress:

### Initial Validation
- [ ] All patches applied without syntax errors
- [ ] Python compilation successful for all files
- [ ] Imports work correctly

### Unit Tests
- [ ] `nef_utils.py` functions pass tests
- [ ] r^-6 averaging works correctly
- [ ] Restraint validation works
- [ ] Multi-assignment parsing works
- [ ] Restraint filtering works

### Integration Tests
- [ ] NEF file parsing works
- [ ] Bayesian steering forward pass works
- [ ] Time-dependent variance works correctly
- [ ] Single-batch training runs without crashes
- [ ] Loss values are reasonable (not NaN/Inf)

### Loss and Scheduling
- [ ] Bayesian loss appears in logs
- [ ] Loss normalization verified
- [ ] Different steering schedules work
- [ ] Schedule visualization created

### Training Tests
- [ ] Multi-batch training stable
- [ ] Full fine-tuning completes
- [ ] Checkpoints save correctly
- [ ] Logging works properly

### Inference Tests
- [ ] Inference with steering works
- [ ] Output structures are valid
- [ ] Restraints are satisfied better than baseline
- [ ] Performance comparison completed

### Final Validation
- [ ] Results are reproducible
- [ ] Documentation updated
- [ ] Code ready for production

---

## Common Issues & Solutions

### Issue 1: KeyError: 'atom_to_idx_list'

**Error message:**
```
KeyError: 'atom_to_idx_list'
```

**Cause:** Patch #5 (datamodule_utils.py) not applied correctly.

**Solution:**
1. Re-check `src/simplefold/utils/datamodule_utils.py`
2. Ensure these lines exist at the end of `collate()`:
```python
if 'atom_to_idx' in keys:
    collated['atom_to_idx_list'] = [d['atom_to_idx'] for d in data]
if 'atom_names' in keys:
    collated['atom_names_list'] = [d['atom_names'] for d in data]
if 'bonds' in keys:
    collated['bonds_list'] = [d['bonds'] for d in data]
```

---

### Issue 2: Bayesian Loss is NaN

**Error message:**
```
loss/bayesian: nan
```

**Possible causes:**
1. Invalid atom indices
2. Zero distances
3. Gradient issues
4. Invalid restraints

**Debug steps:**

```python
# Add debugging in bayesian_steering.py calculate_enoe()
print(f"atom1_indices: {atom1_indices}")
print(f"atom2_indices: {atom2_indices}")
print(f"dist_sq: {dist_sq}")
print(f"dist_minus_6: {dist_minus_6}")

# Check for invalid values
assert not torch.isnan(dist_sq).any(), "NaN in distances"
assert not torch.isinf(dist_minus_6).any(), "Inf in r^-6"
```

**Solution:**
- Verify atom mappings are correct
- Check restraint file validity
- Ensure coordinates are finite
- Add epsilon to prevent division by zero

---

### Issue 3: Training Very Slow