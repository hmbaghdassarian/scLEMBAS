# Install from the repository root with:
#   pip install .            (regular install)
#   pip install -e .         (editable / development install)
from setuptools import setup
from setuptools import find_packages


classes = """
    License :: OSI Approved :: MIT License
    Programming Language :: Python :: 3
    Topic :: Software Development :: Libraries
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Bio-Informatics
    Operating System :: Microsoft :: Windows
    Operating System :: Unix
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

DISTNAME = 'scLEMBAS'
AUTHOR = 'Hratch Baghdassarian'
AUTHOR_EMAIL = 'hmbaghdassarian@gmail.com'
DESCRIPTION = 'Python package for multi-context signaling pathway activity prediction at single-cell resolution.'
LICENSE = 'MIT'

VERSION = '0.1.0'
ISRELEASED = False

PYTHON_REQUIRES = '>=3.9'

# Runtime dependencies. These mirror the packages actually imported by the
# scLEMBAS source and are kept consistent with env_setup/env_main.yml.
# NOTE: cuml (RAPIDS) is an optional GPU-only dependency imported lazily inside
# scLEMBAS/_scanpy_umap.py; it is not pip-installable and is therefore omitted.
INSTALL_REQUIRES = [
    'torch>=2.1.0',
    'numpy<2.0',
    'scipy<1.13',
    'numba<0.59',
    'pandas',
    'scikit-learn',
    'scanpy',
    'anndata',
    'decoupler==1.5.0',
    'omnipath',
    'umap-learn',
    'pynndescent',
    'plotnine',
    'seaborn',
    'matplotlib',
    'statsmodels',
    'networkx',
    'leidenalg',
    'kneed',
    'geomloss',
    'cliffs-delta',
    'tqdm',
    'tqdm-joblib',
    'joblib',
    'annotated-types',
]

EXTRAS_REQUIRES = {
    'interactive': ['jupyter', 'ipykernel'],
}

with open('README.md') as f:
    long_description = f.read()

metadata = dict(
    name=DISTNAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    url='https://github.com/hmbaghdassarian/scLEMBAS',  # homepage
    packages=find_packages(include=['scLEMBAS', 'scLEMBAS.*'], exclude=('*test*',)),
    project_urls={'Documentation': 'https://hmbaghdassarian.github.io/scLEMBAS/'},
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRES,
    classifiers=classifiers,
    license=LICENSE,
)


def setup_package() -> None:
    setup(**metadata)


if __name__ == '__main__':
    setup_package()