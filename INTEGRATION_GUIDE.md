# SimpleFold Integration Guide

This guide provides exact code modifications needed to integrate SimpleNOEFold capabilities into SimpleFold.

## 1. Modify simplefold.py - Add Bayesian Loss Support

### Location: `model/simplefold.py` - `__init__()` method

Add these hyperparameters to the `__init__` method (around line 85):

```python
def __init__(
    self,
    architecture,
    processor,
    loss,
    path,
    sampler,
    optimizer=None,
    scheduler=None,
    plddt_module=None,
    ema_decay=0.999,
    esm_model="esm2_3B",
    aa_bolt_link=None,
    use_rigid_align=True,
    smooth_lddt_loss_weight=1.0,
    lddt_cutoff=15.0,
    clip_grad_norm_val=None,
    lddt_weight_schedule=False,
    plddt_training=False,
    sample_dir='artifacts/',
    # NEW: Bayesian loss parameters
    use_bayesian_loss=False,
    bayesian_loss_type="noe_geom",
    noe_potential_type="gaussian",
    noe_base_sigma=0.5,
    noe_time_dependent=True,
    geometric_clash_cutoff=2.5,
    geometric_clash_penalty=10.0,
    geometric_covalent_penalty=5.0,
    bayesian_alpha_schedule="linear",
    bayesian_beta_start=0.1,
    bayesian_beta_end=0.5,
):
    super().__init__()
    self.save_hyperparameters(logger=False)
    
    # ... existing code ...
    
    # NEW: Initialize Bayesian components if needed
    if use_bayesian_loss:
        from energy.noe_energy import NOEEnergy
        from energy.geometric_prior import GeometricPrior
        
        # Initialize with empty restraints (will be set per batch if available)
        self.noe_energy = NOEEnergy(
            restraints=[],
            potential_type=noe_potential_type,
            base_sigma=noe_base_sigma,
            time_dependent=noe_time_dependent,
        )
        
        self.geometric_prior = GeometricPrior(
            clash_cutoff=geometric_clash_cutoff,
            clash_penalty_scale=geometric_clash_penalty,
            covalent_penalty_scale=geometric_covalent_penalty,
        )
    else:
        self.noe_energy = None
        self.geometric_prior = None
```

### Location: `model/simplefold.py` - Add helper method after `__init__`

```python
def _compute_bayesian_weight_schedule(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute time-dependent Bayesian loss weights.
    
    Parameters
    ----------
    t : torch.Tensor
        Time step [B]
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - alpha(t): Weight for LDDT loss
        - beta(t): Weight for Bayesian loss
    """
    if self.hparams.bayesian_alpha_schedule == "linear":
        # LDDT weight increases with time (1 + 8*relu(t-0.5))
        alpha_t = 1.0 + 8.0 * torch.relu(t - 0.5)
    else:
        alpha_t = torch.ones_like(t)
    
    # Bayesian weight schedule: linear ramp from start to end
    beta_t = (
        self.hparams.bayesian_beta_start +
        (self.hparams.bayesian_beta_end - self.hparams.bayesian_beta_start) * t
    )
    
    return alpha_t, beta_t
```

### Location: `model/simplefold.py` - Modify `flow_matching_train_step()` method

Find the section where MSE loss is computed (around line 470). Add Bayesian loss:

**BEFORE (existing code):**
```python
def flow_matching_train_step(self, batch, batch_idx):
    # ... preprocessing ...
    
    loss = F.mse_loss(out_dict['predict_velocity'], target, reduction='none')
    loss_mask = resolved_atom_mask * align_weights
    loss = self.loss_masking(loss, loss_mask)
    loss = loss.mean()
    
    self.log("loss/mse", loss.item(), ...)
    
    if self.use_smooth_lddt_loss:
        # ... smooth LDDT loss code ...
```

**AFTER (with Bayesian loss):**
```python
def flow_matching_train_step(self, batch, batch_idx):
    # ... preprocessing ...
    
    loss = F.mse_loss(out_dict['predict_velocity'], target, reduction='none')
    loss_mask = resolved_atom_mask * align_weights
    loss = self.loss_masking(loss, loss_mask)
    loss = loss.mean()
    
    self.log("loss/mse", loss.item(), ...)
    
    # NEW: Add Bayesian loss if enabled
    if self.hparams.use_bayesian_loss and self.noe_energy is not None:
        # Get time-dependent weights
        alpha_t, beta_t = self._compute_bayesian_weight_schedule(t)
        
        # One-step Euler to get denoised coordinates
        denoised_coords = y_t + out_dict['predict_velocity'] * (1.0 - t[:, None, None])
        
        # Rescale to Angstroms
        denoised_coords = center_random_augmentation(
            denoised_coords,
            batch['atom_pad_mask'],
            augmentation=False,
            centering=True,
        ) * self.processor.scale
        
        # Compute Bayesian energies
        bayesian_loss_total = torch.zeros(1, device=self.device)
        
        if self.hparams.bayesian_loss_type in ["noe_only", "noe_geom"]:
            noe_energy, noe_stats = self.noe_energy(denoised_coords, t)
            noe_loss = torch.mean(beta_t * noe_energy)
            bayesian_loss_total += noe_loss
            
            self.log(
                "loss/noe",
                noe_loss.item(),
                on_epoch=True,
                logger=True,
                prog_bar=False,
                rank_zero_only=True,
            )
            self.log(
                "metric/noe_violations",
                noe_stats["fraction_violated"].item(),
                on_epoch=True,
                logger=True,
                prog_bar=False,
            )
        
        if self.hparams.bayesian_loss_type in ["geom_only", "noe_geom"]:
            geom_energy, geom_stats = self.geometric_prior(
                denoised_coords,
                batch['atom_pad_mask']
            )
            geom_loss = torch.mean(beta_t * geom_energy)
            bayesian_loss_total += geom_loss
            
            self.log(
                "loss/geometric",
                geom_loss.item(),
                on_epoch=True,
                logger=True,
                prog_bar=False,
                rank_zero_only=True,
            )
        
        # Add Bayesian loss to total
        loss = loss + bayesian_loss_total
        
        self.log(
            "loss/bayesian_total",
            bayesian_loss_total.item(),
            on_epoch=True,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
        )
    
    if self.use_smooth_lddt_loss:
        # ... existing smooth LDDT loss code ...
        # (unchanged)
```

### Location: `model/simplefold.py` - Update imports at top of file

Add these imports after the existing imports (around line 25):

```python
# Add to imports section
from energy.noe_energy import NOEEnergy
from energy.geometric_prior import GeometricPrior
```

---

## 2. Modify inference.py - Support NEF-Guided Prediction

### Location: `inference.py` - Add import

```python
from model.torch.steering import BayesianSteeredEMSampler
from inference_nef import BayesianEnergyGradient
from nef.parser import NEFParser
```

### Location: `inference.py` - Modify `predict_structures_from_fastas()` or prediction loop

Add optional NEF file support (pseudo-code):

```python
def predict_with_nef_steering(
    fasta_file: str,
    nef_file: Optional[str] = None,
    num_samples: int = 5,
    use_steering: bool = False,
    steering_weight: float = 1.0,
    output_dir: str = "./output",
):
    """
    Predict structures with optional NMR guidance.
    
    Parameters
    ----------
    fasta_file : str
        Input FASTA file
    nef_file : Optional[str]
        Path to NEF file for NMR-guided prediction
    num_samples : int
        Number of samples to generate
    use_steering : bool
        Whether to use Bayesian steering
    steering_weight : float
        Maximum steering force weight
    output_dir : str
        Output directory
    """
    # Load sequence
    sequences = load_fasta(fasta_file)
    
    # Load model
    model = load_checkpoint(model_ckpt)
    
    # If NEF file provided, set up steering
    steering_fn = None
    if use_steering and nef_file is not None:
        # Parse NEF
        parser = NEFParser(nef_file)
        restraints = parser.get_restraints()
        
        # Create NOE energy from restraints
        noe_distances = [...]  # Convert to NOEDistance objects
        noe_energy = NOEEnergy(noe_distances)
        geometric_prior = GeometricPrior()
        
        # Create energy gradient
        energy_grad = BayesianEnergyGradient(noe_energy, geometric_prior)
        
        # Define steering function
        def steering_fn(coords, t):
            force, stats = energy_grad(coords, t, batch["atom_pad_mask"])
            return force
        
        # Use steered sampler
        sampler = BayesianSteeredEMSampler(
            steering_start_t=0.5,
            steering_max_weight=steering_weight,
        )
    else:
        # Use original sampler
        sampler = model.sampler
    
    # Generate predictions
    for seq in sequences:
        batch = prepare_batch(seq)
        
        for i in range(num_samples):
            noise = torch.randn_like(batch['coords'])
            
            if use_steering and steering_fn is not None:
                output = sampler.sample(
                    model.forward,
                    model.path,
                    noise,
                    batch,
                    steering_fn=steering_fn
                )
            else:
                output = sampler.sample(
                    model.forward,
                    model.path,
                    noise,
                    batch
                )
            
            # Save structure
            save_structure(output, output_dir)
```

---

## 3. Update Config System

### Add to `configs/finetune_bayesian.yaml`:

The file is already created. Just ensure these parameters are set correctly:

```yaml
model:
  _target_: model.simplefold.SimpleFold
  # ... existing params ...
  use_bayesian_loss: True
  bayesian_loss_type: "noe_geom"
  noe_potential_type: "gaussian"
  noe_base_sigma: 0.5
  geometric_clash_cutoff: 2.5
  bayesian_beta_start: 0.1
  bayesian_beta_end: 0.5
```

---

## 4. Testing the Integration

### Test 1: Check imports work

```bash
cd src/simplefold

python -c "
from energy.noe_energy import NOEEnergy
from energy.geometric_prior import GeometricPrior
from model.torch.steering import BayesianSteeredEMSampler
from nef.parser import NEFParser
print('All imports successful!')
"
```

### Test 2: Check Bayesian components initialize

```bash
python -c "
import torch
from energy.noe_energy import NOEEnergy, NOEDistance

# Create test restraints
restraints = [
    NOEDistance([(0, 5)], 4.0),
    NOEDistance([(1, 6)], 4.5),
]

# Initialize NOE energy
noe = NOEEnergy(restraints)

# Test forward pass
coords = torch.randn(2, 20, 3)
t = torch.ones(2)
energy, stats = noe(coords, t)

print(f'Energy shape: {energy.shape}')
print(f'Energy values: {energy}')
print('NOEEnergy test passed!')
"
```

### Test 3: Check gradient computation

```bash
python -c "
import torch
from energy.geometric_prior import GeometricPrior

geom = GeometricPrior()
coords = torch.randn(2, 20, 3, requires_grad=True)
mask = torch.ones(2, 20)

energy, stats = geom(coords, mask)
loss = energy.sum()
loss.backward()

print(f'Gradient shape: {coords.grad.shape}')
print('Gradient computation test passed!')
"
```

---

## 5. Integration Checklist

- [ ] Added Bayesian hyperparameters to `SimpleFold.__init__`
- [ ] Added helper method `_compute_bayesian_weight_schedule`
- [ ] Modified `flow_matching_train_step` to compute Bayesian loss
- [ ] Added imports for energy modules
- [ ] Created/verified `configs/finetune_bayesian.yaml`
- [ ] Tested imports and initialization
- [ ] Tested gradient computation
- [ ] Run small test training with `use_bayesian_loss=True`
- [ ] Verified loss values are logged correctly
- [ ] Created sample NEF file for testing

---

## 6. Example Fine-Tuning Command

```bash
python src/simplefold/train.py \
    --config-name=finetune_bayesian \
    model.load_ckpt_path=/path/to/pretrained/checkpoint.ckpt \
    model.use_bayesian_loss=True \
    model.bayesian_loss_type="noe_geom" \
    model.bayesian_beta_end=0.5 \
    data.train_data_dir=/path/to/training/structures \
    trainer.max_steps=50000 \
    trainer.check_val_every_n_epoch=null
```

---

## Common Issues During Integration

### Issue: Import errors for energy modules

**Solution**: Make sure `energy/` directory exists with `__init__.py` at `src/simplefold/energy/`

### Issue: Bayesian loss NaN values

**Solution**: 
- Check NOE restraint coordinates are within valid ranges
- Reduce `noe_base_sigma` if too loose
- Ensure `beta_t` isn't too large early in training

### Issue: Training slower with Bayesian loss

**Expected**: Adding loss terms increases compute time ~20-30%. Use gradient accumulation if needed.

### Issue: Model diverges during fine-tuning

**Solution**:
- Start with lower `bayesian_beta_end` (e.g., 0.1)
- Use learning rate schedule to decay over time
- Ensure MSE loss hasn't degraded (check loss/mse logs)

---

## Performance Monitoring

Add these to your monitoring:

```python
# In training logs, you should see:
# - loss/mse: MSE between predicted and target velocity (should decrease)
# - loss/noe: NOE energy term (should decrease if restraints present)
# - loss/geometric: Geometric prior energy (may plateau)
# - metric/noe_violations: Fraction of violated restraints (should decrease)
# - loss/smooth_lddt: LDDT-based loss (should decrease)
```

Expected training progression:
- Epoch 1-10: All losses high, rapid decrease
- Epoch 10-100: Steady decrease, starting to plateau
- Epoch 100+: Fine-tuning regime, slow improvement

---

## Next Steps After Integration

1. Test on a small system (< 100 residues)
2. Create validation script to check restraint satisfaction
3. Compare structures with/without steering
4. Measure computational overhead
5. Optimize parameters on your target dataset
