# Virtual Pulse Reconstruction Diagnostic

## Installation

### Create an environment with at least python 3.10

For example using venv

```bash
python3 -m venv vprd-env
source vprd-env/bin/activate
```

### Download the code and install it with pip

```bash
git clone https://github.com/thawn/VPRD
cd VPRD
pip install .
```

## First steps

### Train a model

Execute the notebook docs/notebooks/train_mlp_model.ipynb

### Reproduce data preprocessing

Execute the notebook docs/notebooks/data_merging.ipynb

### Step-by-step image processing

Execute the notebook docs/notebooks/image_to_electron_power
