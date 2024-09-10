# Virtual Pulse Reconstruction Diagnostic

Code for the paper Harnessing Machine Learning for Single-Shot Measurement of Free Electron Laser Pulse Power

## Installation

Download the code and install it with pip

Note: You need at least Python version 3.10

```bash
mkdir VPRD
cd VPRD
curl https://anonymous.4open.science/api/repo/VPRD-1146/zip -o VPRD.zip
unzip VPRD.zip
python3 -m venv vprd-env
source vprd-env/bin/activate
pip install .
```

## First steps

Start jupyter lab (while `vprd-env` is still active):

```bash
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
