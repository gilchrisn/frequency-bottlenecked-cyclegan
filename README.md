# Frequency-Bottlenecked CycleGAN

CS424 Group Project (Task 2) — A zero-parameter frequency bottleneck that prevents steganographic encoding in unpaired medical image translation.

## The Problem

When CycleGAN translates between structurally asymmetric domains (e.g., brain MRI with tumors vs healthy), the generator learns to hide tumor coordinates as imperceptible high-frequency noise rather than faithfully translating. This steganographic encoding satisfies the cycle-consistency loss without producing structurally honest translations.

## Our Solution: FB-CycleGAN

Inject a fixed, differentiable Gaussian low-pass filter into the cycle-consistency path. The blur destroys the high-frequency bandwidth needed for steganographic encoding, forcing the generator to preserve macroscopic pathology or accept an irrecoverable cycle-consistency penalty. Zero additional learnable parameters.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
pip install -r requirements.txt

# Download BraTS 2020 dataset
python scripts/download_data.py

# Preprocess NIfTI volumes -> 2D slices
python scripts/preprocess_brats.py

# View sample images
python scripts/show_samples.py
```

## Project Structure

```
src/
├── config.py              # ALL hyperparameters — no magic numbers elsewhere
├── models/
│   ├── generator.py       # ResNet-9 / ResNet-6 generators
│   ├── discriminator.py   # PatchGAN discriminator (70×70)
│   └── unet.py            # Downstream segmentation U-Net
├── losses/
│   ├── adversarial.py     # GAN losses (LSGAN, vanilla, WGAN-GP)
│   ├── cycle.py           # Cycle-consistency loss
│   ├── identity.py        # Identity loss
│   └── bottleneck.py      # Frequency bottleneck — Gaussian blur in cycle path
├── data/
│   ├── brats_dataset.py   # NIfTI → 2D slice pipeline
│   └── transforms.py      # Augmentation pipelines
├── training/
│   ├── trainer.py         # Training loop (GAN + cycle + identity)
│   ├── scheduler.py       # LR scheduling strategies
│   └── replay_buffer.py   # Image replay buffer for discriminator
├── evaluation/
│   ├── metrics.py         # FID, SSIM, Dice score
│   └── forensics.py       # FFT spectral analysis for steganographic audit
├── visualization/
│   └── plotter.py         # Loss curves, generated images, FFT maps
└── utils.py               # Seeding, logging, parameter counting

scripts/
├── download_data.py       # Download BraTS 2020 from Kaggle
├── preprocess_brats.py    # One-time NIfTI → .npy slice extraction
├── show_samples.py        # Visualize dataset samples
├── analyze_data.py        # Data exploration and statistics
├── forensic_audit.py      # FFT residual maps + perturbation test
├── score.py               # Local evaluation (FID, SSIM, Dice)
└── plot_results.py        # Paper-quality figures

docs/
├── EXPERIMENTS.md         # Run history and scoreboard — LIVING DOCUMENT
├── ARCHITECTURE.md        # Model design and parameter counts — LIVING
└── THEORY.md              # Design decisions and theoretical justification
```

## Running Experiments

```bash
# Train baseline CycleGAN (no bottleneck — control)
python run_experiment.py --preset baseline_cyclegan

# Train FB-CycleGAN with σ=1.0
python run_experiment.py --preset fb_cyclegan_sigma1

# Sigma sweep for Pareto analysis (honesty vs quality)
python sweep_sigma.py --sigmas 0.5 1.0 1.5 2.0 3.0

# Evaluate a checkpoint
python run_experiment.py --eval-only --checkpoint outputs/checkpoints/<name>/best.pt

# Forensic audit — FFT residuals + perturbation test
python scripts/forensic_audit.py --checkpoint outputs/checkpoints/<name>/best.pt
```

## Evaluation Protocol

1. **Steganographic Forensic Audit**: FFT power spectrum of residuals + noise perturbation test
2. **Quality & Fidelity**: FID (generation quality) + SSIM (reconstruction fidelity)
3. **Pareto Analysis**: Sweep σ ∈ {0.5, 1.0, 1.5, 2.0, 3.0} for honesty-quality tradeoff
4. **Downstream Segmentation**: U-Net trained on synthetic images, evaluated on real BraTS (Dice)

## Parallel Workflow (3 contributors)

### Ownership Map

Each person owns a vertical slice of the codebase. You can read anyone's code, but only modify files in your own area. Changes to **shared files** (`src/config.py`, `docs/EXPERIMENTS.md`) require a quick message to the group.

| Area | Owner | Files you OWN | Files you READ but don't modify |
|------|-------|--------------|-------------------------------|
| **Models & Architecture** | Person A | `src/models/`, `src/losses/` | `src/training/trainer.py` (to understand how models are called) |
| **Data & Evaluation** | Person B | `src/data/`, `src/evaluation/`, `src/visualization/`, `scripts/` | `src/models/` (to understand model I/O shapes) |
| **Training & Infra** | Person C | `src/training/`, `run_experiment.py`, `sweep_sigma.py` | `src/models/`, `src/data/` (to wire everything together) |
| **Shared (all)** | Everyone | `src/config.py`, `docs/EXPERIMENTS.md` | — |

### How to add a new generator variant (Person A)

1. Create a new file `src/models/my_generator.py` with your class
2. Have it accept the same `ModelConfig` dataclass as input
3. Register it in `src/models/__init__.py` so the factory can find it
4. Add a matching `generator: str` value in `ModelConfig` (e.g., `"resnet_6blocks"`)
5. **Do NOT modify** `generator.py` — keep the original ResNet-9 untouched

### How to add a new loss function (Person A)

1. Create a new file in `src/losses/` (e.g., `src/losses/perceptual.py`)
2. Have it accept tensors and return a scalar loss
3. Add a config flag in `LossConfig` (e.g., `use_perceptual: bool = False`, `lambda_perceptual: float = 1.0`)
4. Person C wires it into the training loop via the config flag

### How to add a new dataset (Person B)

1. Create a new file in `src/data/` (e.g., `src/data/preprocessed_dataset.py`)
2. Subclass `torch.utils.data.Dataset`, return `{"A": tensor, "B": tensor}` dicts
3. Add any new data config fields to `DataConfig`
4. Person C wires it into the training loop

### How to add a new experiment (Anyone)

1. Add a new preset to `PRESETS` in `src/config.py`
2. Run it: `python run_experiment.py --preset your_preset_name`
3. Log results in `docs/EXPERIMENTS.md` using the template there
4. Update the presets table in `CLAUDE.md` with your scores

### Golden Rules (never break these)

1. **All hyperparameters in `src/config.py`** — no magic numbers anywhere else
2. **New variants = new files/classes** — never modify an existing working model or loss. Extend by adding, not editing (Open-Closed Principle)
3. **One preset per experiment** — every run must be reproducible from a preset name alone
4. **Log every run** in `docs/EXPERIMENTS.md` — even failed ones. Failed experiments are data
5. **Patient-level data splits** — never split by slice. If you change the split, tell everyone
6. **Don't push broken code to main** — use feature branches, merge only when your component works in isolation

### Interfaces Between Components

The three areas communicate through these contracts:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        src/config.py                                │
│  ExperimentConfig = ModelConfig + LossConfig + TrainConfig + ...    │
│  (shared data contract — everyone reads, coordinate before editing) │
└────────────────────┬────────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   src/models/    src/data/    src/training/
   Person A       Person B     Person C
        │            │            │
        │   Model contract:      │
        │   G(x) → tensor same   │
        │   shape as x           │
        │   D(x) → patch map     │
        │            │            │
        └────────────┴────────────┘
                     │
               src/evaluation/
               Person B
```

**Model contract (Person A → Person C):**
- `Generator.forward(x: Tensor[B,C,H,W]) → Tensor[B,C,H,W]` — same shape in, same shape out
- `Discriminator.forward(x: Tensor[B,C,H,W]) → Tensor[B,1,H',W']` — patch classification map
- `count_parameters(model) < 60M` for Task 1 compatibility (not required for Task 2 but good practice)

**Dataset contract (Person B → Person C):**
- `__getitem__` returns `{"A": Tensor[C,H,W], "B": Tensor[C,H,W]}` (pathological, healthy)
- Images normalized to `[-1, 1]` (Tanh output range)
- `__len__` returns dataset size

**Loss contract (Person A → Person C):**
- Each loss function takes tensors, returns a scalar `Tensor` (no `.item()`, must be differentiable)
- Bottleneck `B(x)` takes and returns `Tensor[B,C,H,W]`, differentiable, no learnable params

## Hypotheses

- **H1**: Baseline CycleGAN shows structured high-frequency energy in FFT residuals (steganography confirmed)
- **H2**: FB-CycleGAN shows diffuse, unstructured residuals and degrades gracefully under perturbation
- **H3**: There exists an optimal σ* achieving both acceptable FID and high SSIM (Pareto-optimal)
- **H4**: U-Net trained on FB-CycleGAN synthetic data achieves higher Dice than baseline synthetic data

## Team

| Member | Role |
|--------|------|
| TBD | TBD |
| TBD | TBD |
| TBD | TBD |
