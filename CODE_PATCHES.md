# SimpleNOEFold - Code Patches for Critical Issues

This file contains exact code patches needed to fix all critical issues.

---

## Patch #1: Fix Restraint ID Logic in bayesian_steering.py

**File:** `src/simplefold/model/torch/bayesian_steering.py`  
**Method:** `calculate_enoe()`  
**Lines:** 49-75

### Replace:
```python
    def calculate_enoe(self, xt, restraints, t, atom_to_idx):
        if not restraints or not any(restraints):
            return torch.tensor(0.0, device=xt.device, requires_grad=True)

        batch_size = xt.shape[0]
        device = xt.device

        # Unpack restraints into tensors for vectorized processing
        atom1_indices, atom2_indices, upper_bounds, batch_indices = [], [], [], []
        for i in range(batch_size):
            if not restraints[i]:
                continue

            atom_to_idx_batch = atom_to_idx[i]
            for r in restraints[i]:
                res1, atom1 = r['atom1_residue_number'] - 1, r['atom1_atom_name']
                res2, atom2 = r['atom2_residue_number'] - 1, r['atom2_atom_name']

                if atom1 in atom_to_idx_batch and atom2 in atom_to_idx_batch:
                    atom1_indices.append(atom_to_idx_batch[atom1])
                    atom2_indices.append(atom_to_idx_batch[atom2])
                    upper_bounds.append(r['upper_bound'])
                    batch_indices.append(i)

        if not atom1_indices:
            return torch.tensor(0.0, device=device, requires_grad=True)

        atom1_indices = torch.tensor(atom1_indices, device=device, dtype=torch.long)
        atom2_indices = torch.tensor(atom2_indices, device=device, dtype=torch.long)
        upper_bounds = torch.tensor(upper_bounds, device=device, dtype=torch.float32)
        batch_indices = torch.tensor(batch_indices, device=device, dtype=torch.long)

        atom1_coords = xt[batch_indices, atom1_indices, :]
        atom2_coords = xt[batch_indices, atom2_indices, :]

        dist_sq = ((atom1_coords - atom2_coords) ** 2).sum(dim=-1)
        dist_minus_6 = dist_sq.pow(-3)

        # Group restraints by their ID to perform the r-6 sum
        restraint_ids = [f"{i}_{r['id']}" for i, batch_restraints in enumerate(restraints) for r in batch_restraints]

        # Map string IDs to integers to use with torch.unique
        unique_str_ids = sorted(list(set(restraint_ids)))
        id_map = {str_id: i for i, str_id in enumerate(unique_str_ids)}
        int_restraint_ids = torch.tensor([id_map[rid] for rid in restraint_ids], device=device)

        unique_int_ids, inverse_indices = torch.unique(int_restraint_ids, return_inverse=True)

        # Sum distances for each unique restraint ID
        num_unique_restraints = len(unique_str_ids)
        summed_dist_minus_6 = torch.zeros(num_unique_restraints, device=device).scatter_add_(0, inverse_indices, dist_minus_6)

        effective_dist = summed_dist_minus_6.pow(-1/6)

        # Create a mapping from restraint ID to upper bound
        id_to_upper_bound = {f"{i}_{r['id']}": r['upper_bound'] for i, batch_restraints in enumerate(restraints) for r in batch_restraints}
        # Look up the upper bound for each unique restraint using the original string ID
        unique_upper_bounds = torch.tensor([id_to_upper_bound[str_id] for str_id in unique_str_ids], device=device)

        violations = F.relu(effective_dist - unique_upper_bounds)
        variance = self.get_variance(t)
        enoe = 0.5 * (violations ** 2) / variance

        return enoe.sum()
```

### With:
```python
    def calculate_enoe(self, xt, restraints, t, atom_to_idx):
        if not restraints or not any(restraints):
            return torch.tensor(0.0, device=xt.device, requires_grad=True)

        batch_size = xt.shape[0]
        device = xt.device

        # Unpack restraints into tensors for vectorized processing
        atom1_indices = []
        atom2_indices = []
        upper_bounds = []
        batch_indices = []
        restraint_ids = []
        
        for batch_idx in range(batch_size):
            if not restraints[batch_idx]:  # FIXED: Proper check for None/empty
                continue

            atom_to_idx_batch = atom_to_idx[batch_idx]
            for restraint in restraints[batch_idx]:
                # Handle both single and multi-assignment restraints
                from utils.nef_utils import parse_restraint_assignments
                assignments = parse_restraint_assignments(restraint)
                
                for assignment in assignments:
                    res1 = assignment['atom1_residue_number']
                    res2 = assignment['atom2_residue_number']
                    atom1 = assignment['atom1_atom_name']
                    atom2 = assignment['atom2_atom_name']
                    
                    # Validate indices exist
                    if atom1 not in atom_to_idx_batch or atom2 not in atom_to_idx_batch:
                        continue
                    
                    atom1_indices.append(atom_to_idx_batch[atom1])
                    atom2_indices.append(atom_to_idx_batch[atom2])
                    upper_bounds.append(assignment['upper_bound'])
                    batch_indices.append(batch_idx)
                    # FIXED: Use unique ID combining batch and restraint ID
                    restraint_ids.append(f"batch_{batch_idx}_id_{restraint['id']}")

        if not atom1_indices:
            return torch.tensor(0.0, device=device, requires_grad=True)

        atom1_indices = torch.tensor(atom1_indices, device=device, dtype=torch.long)
        atom2_indices = torch.tensor(atom2_indices, device=device, dtype=torch.long)
        upper_bounds = torch.tensor(upper_bounds, device=device, dtype=torch.float32)
        batch_indices = torch.tensor(batch_indices, device=device, dtype=torch.long)

        atom1_coords = xt[batch_indices, atom1_indices, :]
        atom2_coords = xt[batch_indices, atom2_indices, :]

        dist_sq = ((atom1_coords - atom2_coords) ** 2).sum(dim=-1)
        dist_minus_6 = dist_sq.pow(-3)

        # FIXED: Create proper ID mapping
        unique_str_ids = sorted(list(set(restraint_ids)))
        id_map = {str_id: idx for idx, str_id in enumerate(unique_str_ids)}
        int_restraint_ids = torch.tensor(
            [id_map[rid] for rid in restraint_ids], 
            device=device,
            dtype=torch.long
        )

        # FIXED: Handle case where no unique restraints
        num_unique_restraints = len(unique_str_ids)
        if num_unique_restraints == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        # Sum r^-6 distances for each restraint
        summed_dist_minus_6 = torch.zeros(num_unique_restraints, device=device)
        summed_dist_minus_6.scatter_add_(0, int_restraint_ids, dist_minus_6)

        # Compute effective distance
        effective_dist = summed_dist_minus_6.pow(-1.0/6.0)

        # FIXED: Create proper upper bound mapping
        id_to_upper_bound = {}
        restraint_idx = 0
        for batch_idx in range(batch_size):
            if not restraints[batch_idx]:
                continue
            for restraint in restraints[batch_idx]:
                key = f"batch_{batch_idx}_id_{restraint['id']}"
                # Use mean of all assignments' upper bounds
                from utils.nef_utils import parse_restraint_assignments
                assignments = parse_restraint_assignments(restraint)
                if assignments:
                    ub = sum(a['upper_bound'] for a in assignments) / len(assignments)
                    id_to_upper_bound[key] = ub
        
        unique_upper_bounds = torch.tensor(
            [id_to_upper_bound[str_id] for str_id in unique_str_ids],
            device=device,
            dtype=torch.float32
        )

        # Compute violations with time-dependent variance
        violations = F.relu(effective_dist - unique_upper_bounds)
        variance = self.get_variance(t)
        enoe = 0.5 * (violations ** 2) / variance

        return enoe.sum()
```

---

## Patch #2: Fix Bayesian Loss Normalization in simplefold.py

**File:** `src/simplefold/model/simplefold.py`  
**Method:** `flow_matching_train_step()`  
**Lines:** 490-500

### Replace:
```python
        # Bayesian steering loss for fine-tuning
        if self.bayesian_steering is not None and 'noesy_restraints' in batch and batch['noesy_restraints']:
            atom_to_idx = batch.get('atom_to_idx', {})
            bonds = batch.get('bonds', [])
            atom_names = batch.get('atom_names', [])
            _, e_bayesian = self.bayesian_steering(
                denoised_coords, batch['noesy_restraints'], t, bonds, atom_to_idx, atom_names
            )
            #loss += e_bayesian * self.bayesian_steering.bayesian_loss_weight
            loss += e_bayesian * self.bayesian_loss_weight
            self.log(
                "loss/bayesian",
                #e_bayesian.item() * self.bayesian_steering.bayesian_loss_weight,
                e_bayesian.item() * self.bayesian_loss_weight,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                rank_zero_only=True,
            )
```

### With:
```python
        # Bayesian steering loss for fine-tuning
        if self.bayesian_steering is not None and 'noesy_restraints' in batch and batch['noesy_restraints']:
            # FIXED: Get per-item restraint data
            atom_to_idx_list = batch.get('atom_to_idx_list', [])
            bonds_list = batch.get('bonds_list', [])
            atom_names_list = batch.get('atom_names_list', [])
            
            # Compute Bayesian loss with proper per-batch handling
            total_e_bayesian = torch.tensor(0.0, device=self.device)
            total_restraints = 0
            
            for batch_idx in range(len(batch['noesy_restraints'])):
                restraints_item = batch['noesy_restraints'][batch_idx]
                if not restraints_item:
                    continue
                
                # Extract per-item data
                atom_to_idx_item = atom_to_idx_list[batch_idx] if atom_to_idx_list else {}
                bonds_item = bonds_list[batch_idx] if bonds_list else []
                atom_names_item = atom_names_list[batch_idx] if atom_names_list else []
                
                # Compute steering force for this item
                coords_item = denoised_coords[batch_idx:batch_idx+1]
                restraints_batch = [restraints_item]  # Wrap in list for batch format
                
                _, e_bayesian_item = self.bayesian_steering(
                    coords_item, restraints_batch, t[batch_idx:batch_idx+1],
                    bonds_item, atom_to_idx_item, atom_names_item
                )
                
                total_e_bayesian += e_bayesian_item
                total_restraints += len(restraints_item)
            
            # FIXED: Normalize by number of restraints
            if total_restraints > 0:
                e_bayesian_normalized = total_e_bayesian / total_restraints
                loss += e_bayesian_normalized * self.bayesian_loss_weight
                
                self.log(
                    "loss/bayesian",
                    e_bayesian_normalized.item(),
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                    rank_zero_only=True,
                )
                
                self.log(
                    "loss/bayesian_raw",
                    total_e_bayesian.item(),
                    on_epoch=True,
                    logger=True,
                    prog_bar=False,
                    rank_zero_only=True,
                )
```

---

## Patch #3: Add Steering Schedule to sampler.py

**File:** `src/simplefold/model/torch/sampler.py`  
**Method:** `__init__()` and `steering_schedule()`

### Replace:
```python
    def __init__(
        self,
        num_timesteps=500,
        t_start=1e-4,
        tau=0.3,
        log_timesteps=False,
        w_cutoff=0.99,
        bayesian_steering: BayesianSteering = None,
        steering_schedule_gamma: float = 1.0,
    ):
        # ... rest of init ...

    def steering_schedule(self, t):
        # Time-dependent scaling for the steering force
        return self.steering_schedule_gamma * t
```

### With:
```python
    def __init__(
        self,
        num_timesteps=500,
        t_start=1e-4,
        tau=0.3,
        log_timesteps=False,
        w_cutoff=0.99,
        bayesian_steering: BayesianSteering = None,
        steering_schedule_gamma: float = 1.0,
        steering_schedule_type: str = 'linear',  # ADDED
    ):
        self.num_timesteps = num_timesteps
        self.log_timesteps = log_timesteps
        self.t_start = t_start
        self.tau = tau
        self.w_cutoff = w_cutoff
        self.bayesian_steering = bayesian_steering
        self.steering_schedule_gamma = steering_schedule_gamma
        self.steering_schedule_type = steering_schedule_type  # ADDED

        if self.log_timesteps:
            t = 1.0 - torch.logspace(-2, 0, self.num_timesteps + 1).flip(0)
            t = t - torch.min(t)
            t = t / torch.max(t)
            self.steps = t.clamp(min=self.t_start, max=1.0)
        else:
            self.steps = torch.linspace(
                self.t_start, 1.0, steps=self.num_timesteps + 1
            )

    def steering_schedule(self, t):
        # FIXED: Implement multiple schedule types
        if self.steering_schedule_type == 'linear':
            # Linear ramp: starts at 0, increases to gamma at t=1
            return t * self.steering_schedule_gamma
        
        elif self.steering_schedule_type == 'sigmoid':
            # Sigmoid: smooth ramp centered at t=0.5
            return torch.sigmoid(5 * (t - 0.5)) * self.steering_schedule_gamma
        
        elif self.steering_schedule_type == 'exponential':
            # Exponential: slowly increases early, faster later
            return (torch.exp(t) - 1) / (torch.e - 1) * self.steering_schedule_gamma
        
        elif self.steering_schedule_type == 'constant':
            # Constant: always gamma
            return torch.ones_like(t) * self.steering_schedule_gamma
        
        elif self.steering_schedule_type == 'square':
            # Quadratic: t^2 schedule
            return (t ** 2) * self.steering_schedule_gamma
        
        else:
            raise ValueError(f"Unknown steering schedule type: {self.steering_schedule_type}")
```

---

## Patch #4: Update sampler.py to Fix Batch Indexing

**File:** `src/simplefold/model/torch/sampler.py`  
**Method:** `euler_maruyama_step()`  
**Lines:** 79-86

### Replace:
```python
        f_steering = torch.zeros_like(y)
        if self.bayesian_steering is not None and 'noesy_restraints' in batch and batch['noesy_restraints']:
            # This requires atom_to_idx mapping and bond information to be in the batch
            atom_to_idx = batch.get('atom_to_idx', {})
            bonds = batch.get('bonds', [])
            atom_names = batch.get('atom_names', [])
            f_steering, _ = self.bayesian_steering(
                y, batch['noesy_restraints'], t, bonds, atom_to_idx, atom_names
            )
```

### With:
```python
        f_steering = torch.zeros_like(y)
        if self.bayesian_steering is not None and 'noesy_restraints' in batch and batch['noesy_restraints']:
            # FIXED: Use per-item metadata from batch
            atom_to_idx_list = batch.get('atom_to_idx_list', [])
            bonds_list = batch.get('bonds_list', [])
            atom_names_list = batch.get('atom_names_list', [])
            
            f_steering_list = []
            for batch_idx in range(y.shape[0]):
                restraints_item = batch['noesy_restraints'][batch_idx]
                
                if not restraints_item or not atom_to_idx_list:
                    f_steering_list.append(torch.zeros_like(y[batch_idx]))
                    continue
                
                # Get per-item data
                atom_to_idx_item = atom_to_idx_list[batch_idx]
                bonds_item = bonds_list[batch_idx] if bonds_list else []
                atom_names_item = atom_names_list[batch_idx] if atom_names_list else []
                
                # Compute steering force for this item
                f_steering_item, _ = self.bayesian_steering(
                    y[batch_idx:batch_idx+1],
                    [restraints_item],  # Wrap in list
                    t,
                    bonds_item,
                    atom_to_idx_item,
                    atom_names_item
                )
                
                f_steering_list.append(f_steering_item.squeeze(0))
            
            if f_steering_list:
                f_steering = torch.stack(f_steering_list, dim=0)
```

---

## Patch #5: Update datamodule_utils.py collate() Function

**File:** `src/simplefold/utils/datamodule_utils.py`  
**Method:** `collate()`

### Add to the end of the function:
```python
        # Store per-item metadata for batch processing
        # These are used by Bayesian steering to process restraints per item
        if 'atom_to_idx' in keys:
            # atom_to_idx should remain as list of dicts, not stacked
            collated['atom_to_idx_list'] = [d['atom_to_idx'] for d in data]
        
        if 'atom_names' in keys:
            # atom_names should remain as list of lists
            collated['atom_names_list'] = [d['atom_names'] for d in data]
        
        if 'bonds' in keys:
            # bonds should remain as list of lists
            collated['bonds_list'] = [d['bonds'] for d in data]

        # Keep original stacked versions for backward compatibility if needed
        # collated['atom_to_idx'] = [d.get('atom_to_idx', {}) for d in data]
        # collated['atom_names'] = [d.get('atom_names', []) for d in data]
        # collated['bonds'] = [d.get('bonds', []) for d in data]

    return collated
```

---

## Patch #6: Update base_train.yaml Config

**File:** `configs/base_train.yaml`

### Replace defaults section:
```yaml
defaults:
  - data: pdb
  - model/architecture: foldingdit_100M
  - model/sampler: euler_maruyama
  - model: simplefold
  - model/processor: protein_processor
  - compute: local
  - callbacks: default
  - logger: null
  - trainer: default
  - paths: default
  - extras: default
  - hydra: default
  - _self_
  - experiment: null
```

### With:
```yaml
defaults:
  - data: pdb
  - model/architecture: foldingdit_100M
  - model/sampler: euler_maruyama
  - model/bayesian_steering: default  # ADDED: Instantiate Bayesian steering
  - model: simplefold
  - model/processor: protein_processor
  - compute: local
  - callbacks: default
  - logger: null
  - trainer: default
  - paths: default
  - extras: default
  - hydra: default
  - _self_
  - experiment: null
```

---

## Patch #7: Update euler_maruyama.yaml Config

**File:** `configs/model/sampler/euler_maruyama.yaml`

### Add at end:
```yaml
# Steering schedule parameters
steering_schedule_type: 'linear'  # Options: 'linear', 'sigmoid', 'exponential', 'constant', 'square'
steering_schedule_gamma: 1.0       # Overall scale factor for steering force
```

---

## Summary of Changes

| Patch # | File | Type | Risk |
|---------|------|------|------|
| #1 | bayesian_steering.py | Logic Fix | HIGH - Core calculation |
| #2 | simplefold.py | Loss Normalization | HIGH - Training stability |
| #3 | sampler.py | Feature Addition | MEDIUM - New options |
| #4 | sampler.py | Batch Indexing | HIGH - Correctness |
| #5 | datamodule_utils.py | Collate Update | MEDIUM - Data flow |
| #6 | base_train.yaml | Config | LOW - Just enables module |
| #7 | euler_maruyama.yaml | Config | LOW - Just adds params |

All patches must be applied for the system to work correctly.

---