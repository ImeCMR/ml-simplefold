# SimpleNOEFold Implementation Guide

## Overview

SimpleNOEFold is an extension of SimpleFold that integrates sparse Nuclear Overhauser Effect Spectroscopy (NOESY) NMR data to constrain protein structure predictions. This guide walks you through the implementation, modifications to the codebase, and how to use the system.

## Architecture Overview

### Core Components

1. **Bayesian Energy Functions** (`energy/` module)
   - `NOEEnergy` (`noe_energy.py`): Computes agreement between predicted structures and NOESY restraints
   - `GeometricPrior` (`geometric_prior.py`): Machine-learned structural constraints replacing physics-based force fields

2. **Steering Module** (`model/torch/steering.py`)
   - `BayesianSteeredEMSampler`: Modified Euler-Maruyama solver incorporating external steering forces
   - `SteeringSchedule`: Time-dependent weighting γ(t) for constraint influence

3. **NEF Parser** (`nef/` module)
   - `NEFParser`: Parses NMR Exchange Format files to extract distance restraints
   - Handles ambiguous NOE assignments

4. **Inference Interface** (`inference_nef.py`)
   - `SimpleNOEFoldInference`: High-level API for NEF-guided prediction
   - `BayesianEnergyGradient`: Computes steering forces from energy functions

---

## Mathematical Foundation

### Standard SDE (from SimpleFold)

$$dx_t = v_\theta(x_t, s, t) dt + \sqrt{\tau \cdot \omega(t)} dW_t$$

### Modified SDE with Steering (SimpleNOEFold)

$$dx_t = [v_\theta(x_t, s, t) + \gamma(t)F_{\text{steering}}(x_t)] dt + \sqrt{\tau \cdot \omega(t)} dW_t$$

Where:
- $v_\theta$: Learned velocity field (implicit prior)
- $F_{\text{steering}} = -\nabla_{x_t} E_{\text{Bayesian}}'(x_t)$: Steering force
- $\gamma(t)$: Time-dependent schedule (increases toward t→1)
- $E_{\text{Bayesian}}'(x_t) = E_{NOE}(x_t) + E_{\text{Geom}}(x_t)$: Total Bayesian energy

### NOE Energy: $E_{NOE}$

$$E_{NOE}(X_t) = \sum_i w_i \cdot P(\Delta d_i; \sigma_i(t))$$

Where:
- Effective distance: $d_{\text{eff},i}(x_t) = \left(\sum_k d_{k,i}^{-6}\right)^{-1/6}$
- Violation: $\Delta d_i = d_{\text{eff},i}(x_t) - d_{UB,i}$
- Gaussian potential: $P(\Delta d; \sigma) = \exp\left(-\frac{(\Delta d)^2}{2\sigma^2}\right)$
- Time-dependent variance: $\sigma_i(t) = \sigma_0 \cdot (1 + 5t)$ (loose early, tight later)

### Geometric Prior: $E_{Geom}$

$$E_{\text{Geom}}(X_t) = E_{\text{Clash}}(X_t) + E_{\text{Covalent}}(X_t)$$

- **Clash**: $E_{\text{Clash}} = \sum_{i \neq j} \max(0, d_{\text{cutoff}} - d_{ij})^2$
- **Covalent**: Penalties for deviations in bond lengths, angles from ideal values

---

## Implementation Details

### 1. Bayesian Energy Modules

#### NOEEnergy (`energy/noe_energy.py`)

**Key Classes:**
- `NOEDistance`: Represents a restraint with possible ambiguous assignments
- `NOEEnergy`: Computes E_NOE for given coordinates

**Usage:**
```python
from energy.noe_energy import NOEEnergy, NOEDistance

# Create restraints
restraints = [
    NOEDistance(
        atom_indices=[(0, 5), (0, 6)],  # Ambiguous: atom 0 to either 5 or 6
        upper_bound=4.0,  # Angstroms
        lower_bound=1.8,
        weight=1.0
    ),
    # ... more restraints
]

# Create NOE energy function
noe_energy = NOEEnergy(
    restraints=restraints,
    potential_type="gaussian",
    base_sigma=0.5,
    time_dependent=True
)

# Compute energy and violations
coords = torch.randn(batch_size, n_atoms, 3)
t = torch.ones(batch_size)
energy, stats = noe_energy(coords, t)
```

**Key Methods:**
- `compute_effective_distance()`: Computes r⁻⁶ summed distance
- `gaussian_potential()` / `lognormal_potential()`: Differentiable potentials
- `compute_time_dependent_sigma()`: Time-dependent variance schedule

#### GeometricPrior (`energy/geometric_prior.py`)

**Key Classes:**
- `GeometricPrior`: Computes E_Geom with clash and covalent penalties
- `CovalentGeometry`: Empirical bond length/angle data

**Usage:**
```python
from energy.geometric_prior import GeometricPrior

geom_prior = GeometricPrior(
    clash_cutoff=2.5,
    clash_penalty_scale=10.0,
    covalent_penalty_scale=5.0,
)

energy, stats = geom_prior(coords, atom_mask)
```

### 2. Steering Module

#### BayesianSteeredEMSampler (`model/torch/steering.py`)

**Modified Euler-Maruyama step:**

```python
# Original velocity
velocity = model_fn(y_t, s, t)

# Steering force (computed externally)
steering_force = -∇_{x_t} E_Bayesian(x_t)

# Time-dependent weight
gamma = steering_schedule(t)

# Modified drift
drift = velocity + diff_coeff * score + gamma * steering_force
```

**Usage:**
```python
from model.torch.steering import BayesianSteeredEMSampler

sampler = BayesianSteeredEMSampler(
    num_timesteps=500,
    steering_schedule_type="linear",
    steering_start_t=0.5,
    steering_max_weight=1.0,
)

# During inference, provide steering function
def compute_steering_force(coords, t):
    energy_grad = compute_bayesian_gradient(coords, t)
    return -energy_grad  # Negative gradient

output = sampler.sample(
    model_fn,
    flow,
    noise,
    batch,
    steering_fn=compute_steering_force
)
```

**Steering Schedule Options:**
- `"linear"`: γ(t) = 0 for t < start_t, then linear ramp to γ_max
- `"sigmoid"`: Smooth sigmoid transition
- `"constant"`: Step function (0 or max)

### 3. NEF Parser

#### NEFParser (`nef/parser.py`)

Parses NMR Exchange Format files (XML-based) to extract distance restraints.

**Usage:**
```python
from nef.parser import NEFParser

parser = NEFParser("experiment.nef")
restraints = parser.get_restraints()

for restraint in restraints:
    print(f"{restraint.atom1} - {restraint.atom2}: {restraint.upper_bound} Å")

print(parser.summary())
```

**Supported Features:**
- XML parsing with namespace handling
- Fallback text-based parsing
- Handles ambiguous NOE assignments
- Extracts upper/lower bounds and weights

---

## Integration with SimpleFold

### 1. Modified Training Loss

Update `SimpleFold.flow_matching_train_step()` to include Bayesian terms:

```python
# Original FM loss
mse_loss = F.mse_loss(predicted_velocity, target_velocity)

# Add Bayesian loss during fine-tuning
if use_bayesian_loss:
    # Get current structure prediction
    denoised_coords = y_t + predicted_velocity * (1.0 - t[:, None, None])
    
    # Compute Bayesian energies
    noe_energy, noe_stats = noe_energy_fn(denoised_coords, t)
    geom_energy, geom_stats = geometric_prior(denoised_coords, atom_mask)
    
    # Time-dependent weights
    alpha_t = schedule_alpha(t)  # LDDT weight
    beta_t = schedule_beta(t)    # Bayesian weight
    
    # Combined loss
    loss = mse_loss + alpha_t * lddt_loss + beta_t * (noe_energy + geom_energy)
```

### 2. Modified Sampler Usage

During inference with NEF data:

```python
# Create energy gradient function
energy_grad = BayesianEnergyGradient(noe_energy, geometric_prior)

def steering_fn(coords, t):
    force, stats = energy_grad(coords, t, atom_mask)
    return force

# Use steered sampler instead of standard sampler
sampler = BayesianSteeredEMSampler(...)
output = sampler.sample(
    model_fn,
    flow,
    noise,
    batch,
    steering_fn=steering_fn
)
```

---

## Fine-Tuning Workflow

### Step 1: Prepare Fine-Tuning Data

Gather high-quality PDB structures with associated NMR data (NEF files):
- Preferably structures < 20 kDa (computational efficiency)
- High-resolution X-ray or cryo-EM structures as reference
- NOESY data from solution NMR experiments

### Step 2: Create NEF Files

If you have raw NOESY peak lists, convert to NEF format:
```bash
# Using CCPN software or similar
python scripts/peaks_to_nef.py --input peaks.txt --output structure.nef
```

### Step 3: Fine-Tune Model

```bash
cd ml-simplefold

# Basic fine-tuning
python src/simplefold/train.py \
    --config-name=finetune_bayesian \
    model.load_ckpt_path=path/to/pretrained/checkpoint.ckpt \
    data.train_data_dir=/path/to/training/data \
    trainer.max_steps=50000

# With custom parameters
python src/simplefold/train.py \
    --config-name=finetune_bayesian \
    model.load_ckpt_path=checkpoint.ckpt \
    model.bayesian_beta_start=0.2 \
    model.bayesian_beta_end=1.0 \
    model.noe_base_sigma=0.3 \
    trainer.max_steps=50000
```

### Step 4: Monitor Training

Watch for:
- **MSE loss**: Should decrease steadily
- **Smooth LDDT loss**: Should decrease
- **Bayesian energy**: Should decrease as model learns constraints
- **Fraction violated**: NOE violations should decrease over time

---

## Inference with NEF Data

### Basic Usage

```python
from inference_nef import SimpleNOEFoldInference

# Initialize inference interface
nef_fold = SimpleNOEFoldInference(
    model_checkpoint="path/to/finetuned/model.ckpt",
    device="cuda",
    esm_model="esm2_3B"
)

# Load fine-tuned model
model = load_model(...)  # Your loading function
nef_fold.load_model(model)

# Predict with NMR guidance
result = nef_fold.predict(
    sequence="MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGNEDDYNPVLNDDTPLEHHHHHH",
    nef_file="experiment.nef",
    num_samples=5,
    steering_weight=1.0,
    steering_start_t=0.5
)

# Access results
coords = result["coords"]  # [num_samples, n_atoms, 3]
```

### Advanced Configuration

```python
# Custom steering schedule
sampler = BayesianSteeredEMSampler(
    num_timesteps=1000,  # More steps for refinement
    steering_schedule_type="sigmoid",
    steering_start_t=0.3,  # Start earlier
    steering_max_weight=2.0,  # Stronger constraints
)

# Custom energy weights
energy_grad = BayesianEnergyGradient(
    noe_energy=noe_energy,
    geometric_prior=geometric_prior,
    noe_weight=1.5,  # Emphasize NMR data
    geom_weight=0.05,  # Light geometric constraints
)
```

---

## Key Parameters and Tuning

### NOE Energy Parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `base_sigma` | 0.5 Å | Initial constraint width |
| `potential_type` | "gaussian" | Energy function shape |
| `time_dependent` | True | Adaptive constraint tightening |

**Tuning guidance:**
- Larger `base_sigma`: Looser constraints, more exploration
- `"lognormal"`: Better for positive violations (d > upper_bound)
- Disable `time_dependent`: Fixed constraint width (less recommended)

### Geometric Prior Parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `clash_cutoff` | 2.5 Å | Minimum non-bonded distance |
| `clash_penalty_scale` | 10.0 | Clash penalty weight |
| `covalent_penalty_scale` | 5.0 | Bond geometry penalty weight |

**Tuning guidance:**
- Increase `clash_cutoff` if structures are too compact
- Increase penalty scales if structures violate chemical rules
- Decrease if constraints are too restrictive

### Steering Schedule Parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `steering_schedule_type` | "linear" | Schedule shape |
| `steering_start_t` | 0.5 | When to apply constraints |
| `steering_max_weight` | 1.0 | Maximum constraint weight |

**Tuning guidance:**
- Start with `steering_start_t=0.5` (high-resolution refinement stage)
- Increase `steering_max_weight` if constraints aren't satisfied
- Use `"sigmoid"` for smooth transitions to prevent artifacts

---

## Validation and Evaluation

### 1. NOE Restraint Satisfaction

```python
# Check fraction of satisfied restraints
violations = stats["noe/violations"]  # [batch, num_restraints]
satisfied = torch.sum(violations <= 0.5)  # Within 0.5 Å tolerance
satisfaction_rate = satisfied / violations.numel()
print(f"Satisfaction rate: {satisfaction_rate:.2%}")
```

### 2. Structural Quality Metrics

```python
# pLDDT confidence
plddt = plddt_module(latent, batch)

# RMSD from reference (if available)
from utils.boltz_utils import weighted_rigid_align
aligned_coords = weighted_rigid_align(ref_coords, pred_coords)
rmsd = torch.sqrt(torch.mean((ref_coords - aligned_coords)**2))
```

### 3. Energy Convergence

```python
# Track energy during sampling
noe_energies = []
geom_energies = []

for t in timesteps:
    noe_e, noe_s = noe_energy(coords_t, t)
    geom_e, geom_s = geometric_prior(coords_t, atom_mask)
    noe_energies.append(noe_e.mean().item())
    geom_energies.append(geom_e.mean().item())

# Should show monotonic decrease
plot(noe_energies, label="NOE Energy")
plot(geom_energies, label="Geometric Energy")
```

---

## Troubleshooting

### Problem: NOE Violations Not Decreasing

**Solutions:**
1. Increase `steering_max_weight`: More forceful constraint application
2. Decrease `steering_start_t`: Apply constraints earlier
3. Decrease `noe_base_sigma`: Tighter effective constraints
4. Check NEF parsing: Verify restraint extraction

### Problem: Structures Too Compact/Extended

**Solutions:**
1. For compact: Reduce `clash_penalty_scale`
2. For extended: Increase `clash_penalty_scale` or increase upper bounds
3. Check if `geometric_prior.use_clash` is appropriate for your data

### Problem: Training Loss Not Decreasing

**Solutions:**
1. Lower learning rate (already reduced for fine-tuning)
2. Reduce `bayesian_beta_end`: Don't weight Bayesian terms too heavily
3. Check data quality: Ensure PDB coordinates are reasonable
4. Verify NEF restraints: Use `parser.summary()` to inspect

### Problem: GPU Memory Issues

**Solutions:**
1. Reduce batch size in data config
2. Reduce `num_timesteps` in sampler
3. Use gradient checkpointing
4. Enable CPU offloading (FSDP already does this)

---

## Citation and References

If you use SimpleNOEFold, please cite:

```bibtex
@article{wang2024simplefold,
  title={SimpleFold: Folding Proteins is Simpler than You Think},
  author={Wang, Yuyang and Lu, Jiarui and Jaitly, Navdeep and others},
  journal={arXiv preprint arXiv:2509.18480},
  year={2025}
}
```

For NEF format details:
```bibtex
@article{gutmanas2015nef,
  title={NMR Exchange Format: a unified and open standard for representation of NMR restraint data},
  author={Gutmanas, A and others},
  journal={Nature Structural & Molecular Biology},
  year={2015}
}
```

---

## Contact and Support

For issues or questions:
1. Check the troubleshooting section above
2. Review configuration parameters
3. Ensure NEF file format is correct
4. Test on small systems first
