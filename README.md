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

Please note that your computer needs a GPU that supports OpenCL in order to execute this notebook.

Execute the notebook docs/notebooks/data_merging.ipynb

### Step-by-step image processing

Please note that your computer needs a GPU that supports OpenCL in order to execute this notebook

Execute the notebook docs/notebooks/image_to_electron_power

## Troubleshooting

In case of Runtime errors in th preprocessing and image processing notebooks, please follow the [pyclesperanto GPU troubleshooting guidelines](https://github.com/clEsperanto/pyclesperanto?tab=readme-ov-file#troubleshooting-graphics-cards-drivers).

In that case the easies way is to create the environment with mamba (so you don't need to install OpenCl manually). Please use these installation instructions instead of the ones given above:

### Installation with mamba

If you don't have mamba installed yet, we recommend to [install miniforge](https://github.com/conda-forge/miniforge)

#### Download the code and create an environment

```bash
git clone https://github.com/thawn/VPRD.git
cd VPRD
mamba env create -f env.yml
mamba activate vprd-env
pip install .
```

**MacOS** users may need to install the following package:

```bash
mamba install -c conda-forge ocl_icd_wrapper_apple
```

**Linux** users may need to install the following package: 

```bash
mamba install -c conda-forge ocl-icd-system
```
