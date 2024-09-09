# Virtual Pulse Reconstruction Diagnostic

Code for the paper Harnessing Machine Learning for Single-Shot Measurement of Free Electron Laser Pulse Power

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

Start jupyter lab:

```bash
source vprd-env/bin/activate # activate the environment
jupyter lab
```

Now you can execute the notebooks to reproduce the results from the paper:

### Train a model

Execute the notebook docs/notebooks/train_mlp_model.ipynb

### Reproduce data preprocessing

Please note that your computer needs a GPU that supports OpenCL in order to execute this notebook

Execute the notebook docs/notebooks/data_merging.ipynb

### Step-by-step image processing

Please note that your computer needs a GPU that supports OpenCL in order to execute this notebook

Execute the notebook docs/notebooks/image_to_electron_power
