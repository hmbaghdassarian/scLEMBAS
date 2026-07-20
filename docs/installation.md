# Installation

scLEMBAS requires Python >= 3.9.

## Option 1 — conda

This builds a self-contained conda environment named `scLEMBAS`.

```bash
git clone https://github.com/hmbaghdassarian/scLEMBAS.git
cd scLEMBAS
conda env create -f env_setup/env_main.yml
conda activate scLEMBAS
pip install .
```

Verify the install:

```bash
python -c "import scLEMBAS; print('scLEMBAS installed')"
```

## Option 2 — pip (from GitHub)

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

## Optional extras

Interactive extras (Jupyter kernel support):

```bash
pip install ".[interactive]"
```

## GPU acceleration

`cuml` (RAPIDS) is an optional GPU-only dependency, imported lazily by
`scLEMBAS._scanpy_umap`. It is not pip-installable and is therefore not listed
as a runtime requirement. Install it through conda if you want GPU-accelerated
UMAP; scLEMBAS falls back to the CPU implementation when it is absent.