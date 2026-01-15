# SimpleNOEFold Implementation - Quick Fix Summary

## Status: ✅ Architecture Sound | ⚠️ 6 Critical Bugs Found

---

## 🔴 CRITICAL ISSUES (Must Fix Before First Run)

### Issue #1: Missing File ✅ CREATED
- **File:** `src/simplefold/utils/nef_utils.py` 
- **Status:** ✅ Now created with utility functions
- **Functions provided:**
  - `calculate_effective_distance_r6()` - r^-6 distance averaging
  - `validate_restraint()` - restraint validation
  - `filter_nef_restraints()` - batch filtering
  - `parse_restraint_assignments()` - multi-assignment handling
  - `group_restraints_by_id()` - ID-based lookup
  - `check_restraint_consistency()` - statistics/validation

---

### Issue #2: Restraint ID Logic Bug
**File:** `src/simplefold/model/torch/bayesian_steering.py` (lines 49-75)

**Problem:**
```python
# BROKEN: String ID creation doesn't handle empty restraints
restraint_ids = [f"{i}_{r['id']}" for i, batch_restraints in enumerate(restraints) 
                 for r in batch_restraints]  # Fails if restraints[i] is None!
```

**Fix Required:**
```python
# Add safety check for None/empty restraints
for batch_idx in range(batch_size):
    if not restraints[batch_idx]:  # Skip if None or empty
        continue
    for restraint in restraints[batch_idx]:
        # Create ID mapping here
```

**Affected Methods:**
- `calculate_enoe()` - Entire r^-6 summation logic needs review

---

### Issue #3: No Multi-Assignment Support
**File:** `src/simplefold/datasets/nef_parser.py`

**Problem:** NEF files support ambiguous restraints (multiple possible atom pairs), but your parser only stores single assignments.

**Current:**
```python
restraint = {
    'atom1_residue_number': int(row['sequence_code_1']),
    'atom1_atom_name': row['atom_name_1'],
    # ... only stores ONE assignment
}
```

**Required (with nef_utils helper):**
```python
from utils.nef_utils import parse_restraint_assignments

# Group by restraint_id to collect all assignments
restraint = {
    'id': restraint_id,
    'assignments': [
        {
            'atom1_residue_number': ...,
            'atom1_atom_name': ...,
            'atom2_residue_number': ...,
            'atom2_atom_name': ...,
            'upper_bound': ...,
        },
        # ... multiple assignments support ...
    ]
}
```

---

### Issue #4: Atom Index Mismatch
**File:** `src/simplefold/datasets/train_datamodule.py` (lines 230-235)

**Problem:** NEF atoms are named differently than Boltz tokenizer atoms.
- NEF: `"HD1@TYR"`, `"QA@MET"` (standard PDB naming)
- Boltz: Uses internal naming from tokenized format
- Simple dict lookup will fail

**Current Code (BROKEN):**
```python
atom_to_idx = {atom_name: i for i, atom_name in enumerate(atom_names)}
# This assumes atom_names match NEF naming, which they don't!
```

**Required Fix:**
```python
# Need to map NEF atom names to tokenized indices
# This requires:
# 1. Knowing the tokenized atom naming convention
# 2. Mapping PDB 3-letter residue codes to single-letter codes
# 3. Building residue-specific atom indices

def build_atom_index_from_tokenized(tokenized):
    """Map NEF-style atom names to tokenized indices"""
    atom_to_idx = {}
    for atom_idx, atom_name in enumerate(tokenized.atom_names):
        # Parse tokenized atom_name format
        # Convert to NEF-compatible format
        nef_name = convert_to_nef_format(atom_name)
        atom_to_idx[nef_name] = atom_idx
    return atom_to_idx
```

---

### Issue #5: Steering Schedule Not Implemented
**File:** `src/simplefold/model/torch/sampler.py` (line 100)

**Current (INCOMPLETE):**
```python
def steering_schedule(self, t):
    return self.steering_schedule_gamma * t  # Fixed multiplier, no real schedule
```

**Problem:** No configuration for schedule type, and parameter is not exposed in config.

**Required Fix:**
1. Add to `sampler.__init__()`:
```python
self.steering_schedule_type = kwargs.get('steering_schedule_type', 'linear')
self.steering_schedule_gamma = kwargs.get('steering_schedule_gamma', 1.0)
```

2. Implement multiple schedule options:
```python
def steering_schedule(self, t):
    if self.steering_schedule_type == 'linear':
        return t * self.steering_schedule_gamma
    elif self.steering_schedule_type == 'sigmoid':
        return torch.sigmoid(5 * (t - 0.5)) * self.steering_schedule_gamma
    elif self.steering_schedule_type == 'exponential':
        return (torch.exp(t) - 1) / (torch.e - 1) * self.steering_schedule_gamma
    else:
        raise ValueError(f"Unknown schedule: {self.steering_schedule_type}")
```

3. Add to `configs/model/sampler/euler_maruyama.yaml`:
```yaml
steering_schedule_type: 'linear'  # or 'sigmoid', 'exponential'
steering_schedule_gamma: 1.0
```

---

### Issue #6: Bayesian Loss Not Normalized
**File:** `src/simplefold/model/simplefold.py` (lines 490-500)

**Problem:** Loss sums over all restraints but doesn't normalize. Training becomes unstable with different dataset sizes.

**Current (BROKEN):**
```python
_, e_bayesian = self.bayesian_steering(denoised_coords, batch['noesy_restraints'], ...)
loss += e_bayesian * self.bayesian_loss_weight  # Not normalized!
```

**Fix:**
```python
_, e_bayesian = self.bayesian_steering(denoised_coords, batch['noesy_restraints'], ...)

# Normalize by number of restraints
num_restraints = sum(len(r) for r in batch.get('noesy_restraints', []) if r)
if num_restraints > 0:
    e_bayesian = e_bayesian / num_restraints
    loss += e_bayesian * self.bayesian_loss_weight
    self.log("loss/bayesian", e_bayesian.item(), ...)
```

---

## 🟡 IMPORTANT WARNINGS

### Warning #7: Batch Restraint Indexing
**File:** `src/simplefold/model/torch/sampler.py` (lines 79-86)

**Problem:** Passing batch-level data to per-item steering function.

**Current:**
```python
atom_to_idx = batch.get('atom_to_idx', {})  # This is batched!
# But bayesian_steering expects per-item atom_to_idx
```

**Fix:** Store per-item in dataloader collate:
```python
# In utils/datamodule_utils.py collate():
collated['atom_to_idx_list'] = [d['atom_to_idx'] for d in data]
collated['atom_names_list'] = [d['atom_names'] for d in data]
collated['bonds_list'] = [d['bonds'] for d in data]

# In sampler.py:
for batch_idx in range(batch_size):
    atom_to_idx_item = batch['atom_to_idx_list'][batch_idx]
    # ... use per-item data ...
```

---

### Warning #8: Config Instantiation Missing
**File:** `configs/base_train.yaml`

**Problem:** `bayesian_steering` is `null` - never instantiated.

**Fix:** Update `base_train.yaml`:
```yaml
defaults:
  - ...
  - model/bayesian_steering: default  # ADD THIS LINE
  - ...

model:
  bayesian_steering: ???  # Will be overridden by defaults
```

---

## 📋 Implementation Checklist

- [ ] **Issue #1:** ✅ DONE - Created `nef_utils.py`
- [ ] **Issue #2:** Fix restraint ID logic in `bayesian_steering.py`
- [ ] **Issue #3:** Implement multi-assignment support in `nef_parser.py`
- [ ] **Issue #4:** Fix atom name mapping between NEF and Boltz
- [ ] **Issue #5:** Implement steering schedule in `sampler.py`
- [ ] **Issue #6:** Normalize Bayesian loss in `simplefold.py`
- [ ] **Warning #7:** Fix batch indexing in dataloader
- [ ] **Warning #8:** Fix config defaults for bayesian_steering

---

## 🧪 Recommended Test Order

```bash
# 1. Test NEF parser
pytest tests/test_nef_parser.py

# 2. Test Bayesian steering module
pytest tests/test_bayesian_steering.py

# 3. Test datamodule with NEF
pytest tests/test_datamodule_with_nef.py

# 4. Test sampler with steering
pytest tests/test_sampler_steering.py

# 5. Training test (single batch)
python train.py +experiment=test_nef_finetuning \
  trainer.max_steps=1 \
  trainer.limit_train_batches=1
```

---

## 🚀 Quick Start After Fixes

```bash
# 1. Apply all fixes above
# 2. Prepare NEF files in: data/nef_files/

# 3. Create fine-tuning config
cp configs/base_train.yaml configs/finetune_nef.yaml
# Edit to add: nef_dir: ${paths.data_dir}nef_files/

# 4. Start fine-tuning
python train.py --config-name=finetune_nef \
  model.bayesian_loss_weight=0.1 \
  data.nef_dir=./data/nef_files
```

---

## 📊 Files Modified Summary

| File | Status | Issues |
|------|--------|--------|
| [nef_utils.py](../utils/nef_utils.py) | ✅ Created | None |
| [nef_parser.py](../datasets/nef_parser.py) | ⚠️ Review | Issue #3 |
| [bayesian_steering.py](../model/torch/bayesian_steering.py) | ⚠️ Review | Issue #2 |
| [sampler.py](../model/torch/sampler.py) | ⚠️ Review | Issues #5, #7 |
| [train_datamodule.py](../datasets/train_datamodule.py) | ⚠️ Review | Issues #4, #7 |
| [simplefold.py](../model/simplefold.py) | ⚠️ Review | Issue #6 |
| [datamodule_utils.py](../utils/datamodule_utils.py) | ⚠️ Review | Warning #7 |
| [base_train.yaml](../../configs/base_train.yaml) | ⚠️ Update | Warning #8 |

---

## 📖 Full Review Document

See: [IMPLEMENTATION_REVIEW.md](./IMPLEMENTATION_REVIEW.md)

Contains:
- Detailed explanations of each issue
- Code examples and fixes
- Architecture validation
- Testing strategy
- Configuration examples

