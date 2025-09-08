mamba install -y pytorch conda-forge::jax conda-forge::scvi-tools=1.2 conda-forge::optax=0.2.0 conda-forge::ipykernel 

# optional for using scvi script
mamba install -y conda-forge::scanpy
pip install cliffs-delta geomloss kneed

#pip install anndata=10.0.1