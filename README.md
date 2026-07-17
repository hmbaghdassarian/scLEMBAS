# scLEMBAS

scLEMBAS is a single-cell extension of LEMBAS ([ref 1](https://doi.org/10.1038/s41467-022-30684-y), [ref 2](https://doi.org/10.1101/2025.10.24.684155)), enabling context aware computational modeling of signaling pathway activity in response to perturbation.

## Installation

scLEMBAS requires Python >= 3.9. 

### Option 1 — conda 

This builds a self-contained conda environment named `scLEMBAS`.

```bash
git clone https://github.com/hmbaghdassarian/scLEMBAS.git
cd scLEMBAS
conda env create -f env_setup/env_main.yml
conda activate scLEMBAS
```

Verify the install:

```bash
python -c "import scLEMBAS; print('scLEMBAS installed')"
```

### Option 2 — pip (from GitHub)

Install directly from the repository into your current (activated) Python
environment:

```bash
git clone https://github.com/hmbaghdassarian/scLEMBAS.git
cd scLEMBAS
pip install .
```

For a development (editable) install, use `pip install -e .` instead.

You can also install without cloning first:

```bash
pip install git+https://github.com/hmbaghdassarian/scLEMBAS.git
```

Optional interactive extras (Jupyter kernel support):

```bash
pip install ".[interactive]"
```

## Reference

scLEMBAS is extensively described in manuscript XXX. It can be cited here:

<!-- TODO: Add citation here -->