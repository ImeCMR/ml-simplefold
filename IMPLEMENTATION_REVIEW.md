# SimpleNOEFold Implementation Review

## Executive Summary

Your implementation of SimpleNOEFold is **architecturally sound** with the core concepts properly integrated. However, there are **6 critical issues** and **8 warnings/improvements** that need addressing before production use.

---

## Critical Issues (Must Fix)

### 1. ⚠️ **MISSING: `nef_utils.py` File**
**Severity:** CRITICAL  
**Location:** `src/simplefold/utils/nef_utils.py`  
**Status:** MISSING (not in codebase)

**Problem:**
You referenced creating `nef_utils.py` in your plan, but this file doesn't exist in the utils directory. The NEF parsing logic is currently only in `nef_parser.py`.

**Impact:** 
- No utility functions for NEF file processing
- No helper functions for distance calculation
- Code organization is incomplete

**Solution:**
Create [src/simplefold/utils/nef_utils.py](src/simplefold/utils/nef_utils.py) with helper functions:
```python
# nef_utils.py should contain:
def parse_restraint_assignments(restraint):
    """Extract all possible atom pair assignments from ambiguous restraint"""
    pass

def calculate_effective_distance_r6(distances_list):
    """Compute r^-6 summed effective distance"""
    pass

def filter_nef_restraints(restraints, sequence_length, min_distance=None):
    """Filter out invalid restraints (e.g., invalid residue indices)"""
    pass
```

---

### 2. ⚠️ **CRITICAL BUG: Incorrect `calculate_enoe()` Restraint ID Logic**
**Severity:** CRITICAL  
**Location:** [src/simplefold/model/torch/bayesian_steering.py](src/simplefold/model/torch/bayesian_steering.py#L49-L75)  
**Lines:** 49-75

**Problem:**
```python
restraint_ids = [f"{i}_{r['id']}" for i, batch_restraints in enumerate(restraints) 
                 for r in batch_restraints]
```

This creates string restraint IDs but doesn't handle the case where `restraints[i]` is `None` or empty. More critically, **the r^-6 summation assumes multiple assignments per restraint**, but your current logic doesn't properly track which distances belong to which restraint.

**Current Logic Flaw:**
- You sum all `dist_minus_6` values by unique restraint ID
- But if multiple atoms are assigned to the same restraint, you need to keep them grouped
- Your `id_to_upper_bound` lookup will fail if restraint IDs don't match

**Fix Required:**
```python
def calculate_enoe(self, xt, restraints, t, atom_to_idx):
    # ... setup code ...
    
    # Structure: list of restraints per batch item
    # Each restraint should have multiple possible atom assignments
    assignments = []  # List of (batch_idx, atom1_idx, atom2_idx, upper_bound, restraint_id)
    
    for batch_idx in range(batch_size):
        if not restraints[batch_idx]:
            continue
            
        for restraint in restraints[batch_idx]:
            # Each restraint can have multiple assignments!
            for assignment in restraint.get('assignments', [restraint]):
                atom1_res = assignment['atom1_residue_number'] - 1
                atom1_name = assignment['atom1_atom_name']
                atom2_res = assignment['atom2_residue_number'] - 1
                atom2_name = assignment['atom2_atom_name']
                
                key = f"{atom1_name}_{atom1_res}_{atom2_name}_{atom2_res}"
                if key in atom_to_idx[batch_idx]:
                    assignments.append({
                        'batch_idx': batch_idx,
                        'atom1_idx': atom_to_idx[batch_idx][key][0],
                        'atom2_idx': atom_to_idx[batch_idx][key][1],
                        'upper_bound': assignment['upper_bound'],
                        'restraint_id': restraint['id']
                    })
    
    # ... rest of logic with proper grouping ...
```

---

### 3. ⚠️ **MISSING IMPLEMENTATION: Ambiguous Restraint Handling**
**Severity:** CRITICAL  
**Location:** [src/simplefold/datasets/nef_parser.py](src/simplefold/datasets/nef_parser.py)  
**Issue:** No multi-assignment support

**Problem:**
NEF files support **ambiguous assignments** - one NOE peak can correspond to multiple atom pairs. Your parser only stores single assignments per restraint:

```python
restraint = {
    'id': row['restraint_id'],
    'atom1_residue_number': int(row['sequence_code_1']),
    'atom1_atom_name': row['atom_name_1'],
    'atom2_residue_number': int(row['sequence_code_2']),
    'atom2_atom_name': row['atom_name_2'],
    'upper_bound': float(row['upper_limit']),
}
```

This assumes 1:1 mapping, but NEF allows multiple "or" assignments per restraint.

**Solution:**
Modify `nef_parser.py` to handle multiple assignments:
```python
# In NEFParser.get_distance_restraints():
# Group rows by restraint_id to handle multiple assignments
restraint = {
    'id': row['restraint_id'],
    'assignments': [
        {
            'atom1_residue_number': int(row['sequence_code_1']),
            'atom1_atom_name': row['atom_name_1'],
            'atom2_residue_number': int(row['sequence_code_2']),
            'atom2_atom_name': row['atom_name_2'],
            'upper_bound': float(row['upper_limit']),
        },
        # ... more assignments ...
    ]
}
```

---

### 4. ⚠️ **CRITICAL ERROR: Atom Index Mismatch**
**Severity:** CRITICAL  
**Location:** [src/simplefold/datasets/train_datamodule.py#L230-235](src/simplefold/datasets/train_datamodule.py#L230-235)

**Problem:**
```python
atom_names = tokenized.atom_names
atom_to_idx = {atom_name: i for i, atom_name in enumerate(atom_names)}
bonds = tokenized.bonds
features['atom_names'] = atom_names
features['atom_to_idx'] = atom_to_idx
features['bonds'] = bonds
```

**Issue:** `atom_to_idx` assumes a simple name → index mapping. But:
1. NEF files use standard atom naming (e.g., "HD1@TYR", "QA@MET")
2. `tokenized.atom_names` uses a different convention from the Boltz pipeline
3. The mapping needs to be **position + residue based**, not just name-based

**Example problem:**
- NEF: `"N-CA"` distance between backbone N and CA
- Your mapping: only stores `"N": 0, "CA": 1` globally
- But which N and CA? There's one per residue!

**Required Fix:**
```python
# Proper residue-aware mapping
atom_to_idx = {}
for i, atom_name in enumerate(tokenized.atom_names):
    # Parse atom_name to get residue and atom info
    # Format might be "RES_IDX_ATOM_NAME" from boltz
    atom_to_idx[atom_name] = i

# More importantly, in bayesian_steering.py:
# Match NEF atom names to tokenized atom names
def map_nef_atoms_to_indices(nef_atom_name, residue_num, atom_to_idx, tokenized):
    # Convert NEF naming convention to tokenized naming convention
    pass
```

---

### 5. ⚠️ **MISSING: Steering Schedule Implementation**
**Severity:** CRITICAL  
**Location:** [src/simplefold/model/torch/sampler.py#L100-101](src/simplefold/model/torch/sampler.py#L100-101)

**Current Code:**
```python
def steering_schedule(self, t):
    # Time-dependent scaling for the steering force
    return self.steering_schedule_gamma * t
```

**Problem:**
Your plan states: "$\gamma(t)$ is typically scheduled to increase as $t \rightarrow 1$", meaning the steering force should be **weak early** and **strong late**. 

Your current formula `γ(t) * t` does this, but:
1. **No parameter exposed** - `steering_schedule_gamma` is set to a fixed value, not a schedule
2. **No configuration** - Users can't customize the schedule in YAML
3. **Questionable schedule** - `t * γ` means γ starts at 0 and only reaches max at t=1

**Correct approach (typically used):**
```python
def steering_schedule(self, t):
    # Schedule: loose early, rigorous late
    # Option 1: Linear ramp
    return torch.clamp(t, min=0.0) * self.steering_schedule_gamma
    
    # Option 2: Sigmoid (smoother)
    return torch.sigmoid(5 * (t - 0.5)) * self.steering_schedule_gamma
    
    # Option 3: Exponential
    return (torch.exp(t) - 1) / (torch.e - 1) * self.steering_schedule_gamma
```

**Fix:** Add to sampler initialization and config.

---

### 6. ⚠️ **BUG: Bayesian Loss Not Scaled to Batch**
**Severity:** CRITICAL  
**Location:** [src/simplefold/model/simplefold.py#L490-500](src/simplefold/model/simplefold.py#L490-500)

**Current Code:**
```python
_, e_bayesian = self.bayesian_steering(
    denoised_coords, batch['noesy_restraints'], t, bonds, atom_to_idx, atom_names
)
loss += e_bayesian * self.bayesian_loss_weight
```

**Problem:**
- `e_bayesian` is the **sum** over all restraints: `enoe.sum()`
- This is **NOT normalized by batch size or number of restraints**
- If you have 100 restraints vs 10, the loss scales 10x
- Training will be **unstable and non-reproducible** across different datasets

**Fix:**
```python
_, e_bayesian = self.bayesian_steering(
    denoised_coords, batch['noesy_restraints'], t, bonds, atom_to_idx, atom_names
)
# Normalize by number of valid restraints
num_restraints = sum(len(r) for r in batch.get('noesy_restraints', [[]]) if r)
if num_restraints > 0:
    e_bayesian = e_bayesian / num_restraints
loss += e_bayesian * self.bayesian_loss_weight
```

---

## Important Warnings & Improvements

### 7. ⚠️ **WARNING: Missing Restraint Validation in NEFParser**
**Location:** [src/simplefold/datasets/nef_parser.py#L40-55](src/simplefold/datasets/nef_parser.py#L40-55)

**Current code filters invalid rows:**
```python
df_filtered = df[
    (df['sequence_code_1'] != '.') &
    (df['sequence_code_2'] != '.') &
    (df['upper_limit'] != '.')
]
```

**What's missing:**
- Validate sequence codes are within protein length
- Warn if upper_bound is impossibly small (< 1.5 Å)
- Check for self-restraints (res1 == res2, atom1 == atom2)
- Handle NaN values in upper_limit conversion

**Recommended addition:**
```python
def validate_restraint(restraint, seq_length):
    res1 = restraint['atom1_residue_number']
    res2 = restraint['atom2_residue_number']
    ub = restraint['upper_bound']
    
    if res1 > seq_length or res2 > seq_length:
        return False, f"Residue out of range: {res1} or {res2} > {seq_length}"
    if ub < 1.5:
        return False, f"Upper bound too small: {ub} Å"
    if res1 == res2 and restraint['atom1_atom_name'] == restraint['atom2_atom_name']:
        return False, "Self-restraint"
    return True, None
```

---

### 8. ⚠️ **WARNING: Time-Dependent Variance Mismatch**
**Location:** [src/simplefold/model/torch/bayesian_steering.py#L104-109](src/simplefold/model/torch/bayesian_steering.py#L104-109)

**Current Code:**
```python
def get_variance(self, t):
    if self.time_dependent_variance:
        # Loosely enforce constraints early in the process, and rigorously as t -> 1
        return 1.0 - t + 1e-6
    else:
        return 1.0
```

**Issue:**
- Your comment says "loose early", but `1.0 - t` means variance **increases** as t → 0
- When t=0: variance = 1.0 (maximum, very loose)
- When t=1: variance = 1e-6 (minimum, very strict) ✓ This is correct!
- **The logic is right, but comment is backwards**

**Fix:** Update comment to clarify the variance schedule matches your plan.

---

### 9. ⚠️ **WARNING: bond_angle_penalty May Have Gradient Issues**
**Location:** [src/simplefold/model/torch/bayesian_steering.py#L120-143](src/simplefold/model/torch/bayesian_steering.py#L120-143)

**Current Code:**
```python
cos_angle = (v1 * v2).sum(dim=-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1))
angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
```

**Potential Issues:**
1. **Gradient instability at ±1**: `acos` has zero gradient at cos_angle = ±1
2. **Potential for NaN**: If norm is 0, division produces inf/nan
3. **No masking**: Applies penalty to all angles, including missing atoms

**Better approach:**
```python
def safe_angle_calculation(v1, v2):
    norm1 = torch.norm(v1, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    norm2 = torch.norm(v2, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    cos_angle = (v1 * v2).sum(dim=-1) / (norm1 * norm2).squeeze(-1)
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-6, 1.0 - 1e-6)  # Avoid acos singularities
    return torch.acos(cos_angle)
```

---

### 10. ⚠️ **WARNING: Batch Restraint Handling May Fail**
**Location:** [src/simplefold/model/torch/sampler.py#L79-86](src/simplefold/model/torch/sampler.py#L79-86)

**Current Code:**
```python
if self.bayesian_steering is not None and 'noesy_restraints' in batch and batch['noesy_restraints']:
    atom_to_idx = batch.get('atom_to_idx', {})
    bonds = batch.get('bonds', [])
    atom_names = batch.get('atom_names', [])
```

**Problem:** In a batch of multiple proteins:
- `batch['atom_to_idx']` returns a stacked tensor or list, not per-item
- `batch['atom_names']` is a batch, not per-item
- You're passing **all batch data** to `bayesian_steering()`, but it expects **per-item data**

**The issue:** Your `calculate_enoe()` does try to handle batches, but:
```python
atom_to_idx_batch = atom_to_idx[i]  # This fails! atom_to_idx is not indexable this way
```

**Fix:** Store per-item metadata in dataloader collate:
```python
# In collate():
collated['atom_to_idx_list'] = [d['atom_to_idx'] for d in data]
collated['atom_names_list'] = [d['atom_names'] for d in data]
collated['bonds_list'] = [d['bonds'] for d in data]
```

---

### 11. ⚠️ **WARNING: No Configuration for Bayesian Steering in Inference**
**Location:** Config system

**Issue:**
- `simplefold.yaml` has `bayesian_steering: null`
- There's no way to enable it during fine-tuning or inference
- Config needs proper instantiation of BayesianSteering module

**Solution:**
Update `configs/base_train.yaml` to properly instantiate:
```yaml
model/bayesian_steering: default  # This should be in defaults
# Then in simplefold.yaml:
bayesian_steering:
  _target_: model.torch.bayesian_steering.BayesianSteering
  enoe_weight: ${model.bayesian_steering.enoe_weight}
  egeom_weight: ${model.bayesian_steering.egeom_weight}
  # ... etc
```

---

### 12. ⚠️ **WARNING: Velocity-Based Steering May Not Be Optimal**
**Location:** [src/simplefold/model/torch/sampler.py#L77-85](src/simplefold/model/torch/sampler.py#L77-85)

**Current approach:**
```python
drift = velocity + diff_coeff * score + gamma_t * f_steering
```

**Issue:** You're adding steering to the drift term, which affects the next position estimate. According to your paper:
$$d x_t \approx [v_\theta(x_t, s, t) + \gamma(t) F_{steering}(x_t)] dt + ...$$

This is **correct**, BUT:
- The steering force is computed on **noisy coordinates** $x_t$
- Later coordinates are computed iteratively, so steering errors compound
- **No backtracking mechanism** if steering leads to violations

**Recommendation:** Consider adding:
1. **Projection step**: After steering, project back to valid configurations
2. **Adaptive strength**: Reduce steering if it violates other constraints
3. **Validation**: Check if steering improves or worsens the violations

---

### 13. ⚠️ **WARNING: No Checkpoint/Recovery for NEF-Steered Inference**
**Location:** Inference pipeline

**Issue:**
- If inference fails mid-sampling, there's no way to resume
- NEF restraint violations aren't logged
- Final sample validation is missing

**Recommended:** Add inference checkpoint system:
```python
def sample_with_checkpoints(self, batch, save_every=50):
    for i in range(sampling_timesteps):
        # ... sampling step ...
        if i % save_every == 0:
            save_checkpoint(y_sampled, i, t, batch['record'])
```

---

## Architecture Validation ✓

### What's Correct:

1. **NEF Parser Structure** ✓
   - Correctly reads NEF files using `starfile`
   - Properly filters invalid entries
   - Handles exceptions gracefully

2. **Bayesian Steering Integration** ✓
   - E_NOE and E_Geom properly separated
   - Gradient computation via autograd is correct
   - Module initialization in SimpleFold is present

3. **SDE Sampler Integration** ✓
   - Steering force correctly added to drift term
   - Time scheduling framework exists
   - Bayesian module is called at each step

4. **Datamodule Integration** ✓
   - NEF directory parameter properly passed
   - Restraints loaded and stored in batch
   - Metadata (atom_names, bonds, atom_to_idx) included

5. **Loss Integration** ✓
   - Bayesian loss added to main training loss
   - Weight parameter exposed in config
   - Proper logging of Bayesian loss

---

## Summary of Fixes Required

| Priority | Issue | File | Fix Complexity |
|----------|-------|------|-----------------|
| 🔴 CRITICAL | Missing `nef_utils.py` | `utils/nef_utils.py` | Low |
| 🔴 CRITICAL | Restraint ID logic bug | `bayesian_steering.py#49-75` | Medium |
| 🔴 CRITICAL | No multi-assignment support | `nef_parser.py` | Medium |
| 🔴 CRITICAL | Atom index mismatch | `train_datamodule.py#230-235` | High |
| 🔴 CRITICAL | Steering schedule incomplete | `sampler.py#100` | Low |
| 🔴 CRITICAL | Loss not normalized | `simplefold.py#490-500` | Low |
| 🟡 IMPORTANT | Batch restraint indexing | `sampler.py#79-86` | Medium |
| 🟡 IMPORTANT | Config instantiation | `base_train.yaml` | Low |

---

## Recommended Testing Strategy

### Unit Tests Needed:
```python
# test_nef_parser.py
- Test valid restraint parsing
- Test invalid entry filtering
- Test multi-assignment parsing

# test_bayesian_steering.py
- Test r^-6 summation
- Test time-dependent variance
- Test gradient computation

# test_sampler_with_steering.py
- Test steering force application
- Test schedule evolution
- Test batch handling
```

### Integration Tests:
```python
# test_training_with_nef.py
- Train on single protein with NEF
- Verify Bayesian loss decreases
- Check that violations reduce over time
```

---

## Next Steps

1. **Immediate (Before first run):**
   - Create `nef_utils.py` with helper functions
   - Fix restraint ID logic in `bayesian_steering.py`
   - Fix batch indexing in `sampler.py`
   - Normalize Bayesian loss

2. **Short-term (Before production):**
   - Implement multi-assignment support in NEF parser
   - Fix atom name mapping between NEF and Boltz
   - Add steering schedule configuration
   - Add restraint validation

3. **Medium-term (Optimization):**
   - Implement checkpoint/recovery system
   - Add inference-time validation
   - Profile and optimize steering force computation
   - Add comprehensive logging of restraint violations

---

## Configuration Example (Corrected)

Create a new fine-tuning config: `configs/finetune_with_nef.yaml`

```yaml
# @package _global_

defaults:
  - /data: pdb
  - /model: simplefold
  - /model/bayesian_steering: default  # ADD THIS
  - /model/processor: protein_processor
  - /trainer: fsdp
  - /paths: default
  - /extras: default
  - /hydra: default
  - /logger: tensorboard
  - _self_

# Fine-tuning task
task_name: "finetuning_nef"

seed: 42

# Dataset with NEF files
data:
  train:
    - data_name: "pdb_with_nef"
      tokenized_dir: ${paths.data_dir}pdb/tokenized
      target_dir: ${paths.data_dir}pdb/structures
      nef_dir: ${paths.data_dir}nef_files  # PATH TO NEF FILES
      cropper:
        _target_: boltz_data_pipeline.crop.boltz.BoltzCropper
        # ...

# Model configuration
model:
  bayesian_loss_weight: 0.1  # Weight for Bayesian term
  bayesian_steering:
    enoe_weight: 1.0
    egeom_weight: 1.0
    time_dependent_variance: true

  sampler:
    steering_schedule_gamma: 1.0  # Should add schedule options

# Training
trainer:
  max_steps: 50000
  val_check_interval: 5000
```

---

**End of Review**

