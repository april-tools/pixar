## Environment Setup

We recommand using Conda to manage the environment.


### On Eddie

Here are steps I used to create my environment on Eddie cluster.

Note: Files in /exports/eddie/scratch/ will be automatically cleaned every month. 

1. Connect to eddie and request an interactive session

The login node limits the memory each user can use, which is insufficient to install the environment. We can request an interactive session to do so.

```
ssh xxx@eddie.ecdf.ed.ac.uk
qlogin -pe interactivemem 4 -l h_vmem=4g
```

2. Install conda on Scratch directory

```
cd /exports/eddie/scratch/<UUN>
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p miniconda3
./miniconda3/bin/conda init bash
bash
```

3. Create the environment

The step by step creation of the environment I performed.

```
conda create -n {name} python=3.10
conda activate {name}
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets tokenizers diffusers["torch"]
pip install opencv-python
pip install wandb
conda install -c conda-forge pycairo
conda install -c conda-forge pygobject
conda install -c conda-forge manimpango
conda install -c conda-forge fonttools
pip install pytesseract
pip install editdistance
pip install -U scikit-learn
pip install nltk
pip install apache-beam
```

4. Setup the project

Please make sure you have access to the UOE GitLab first.

```
cd ~/
git clone --recurse-submodules git@git.ecdf.ed.ac.uk:s1891075/msc_project.git
pip install -e pixel
mkdir -p /exports/eddie/scratch/{UUN}/pixelplus
ln -s /exports/eddie/scratch/{UUN}/pixelplus storage
```

## Prepare Dataset

1. Clone the PIXEL weights, fonts

```
git clone https://huggingface.co/Team-PIXEL/pixel-base storage/pixel-base
```