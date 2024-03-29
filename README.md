## Environment Setup

We recommand using Conda to manage the environment.


### On Eddie

Here are steps I used to create my environment on Eddie cluster.

Note: Files in /exports/eddie/scratch/ will be automatically cleaned every month. 

1. Connect to eddie and request an interactive session

The login node limits the memory each user can use, which is insufficient to install the environment. We can request an interactive session to do so.

```
ssh xxx@eddie.ecdf.ed.ac.uk
qlogin -pe interactivemem 16 -l h_vmem=8g
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
conda install -c conda-forge pycairo=1.24.0 cairo=1.16.0 fonttools=4.25.0 pygobject=3.42.1 manimpango=0.4.1 # make sure the version matches, othewise the rendered text may different
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.34.1 datasets==2.14.6 tokenizers==0.14.1 diffusers["torch"]==0.17.1
pip install opencv-python==4.6.0.66 opencv-contrib-python==4.6.0.66
pip install wandb==0.15.12 pytesseract==0.3.10 editdistance==0.6.2 scikit-image==0.22.0 scikit-learn==1.3.0
pip install nltk==3.8.1 apache-beam==2.51.0
pip install paddlepaddle==2.5.2 paddleocr==2.7.0.3
pip install adamr
pip install flash-attn --no-build-isolation # install FlashAttention https://github.com/Dao-AILab/flash-attention
pip install ipywidgets
sudo apt install tesseract-ocr # if you cannot use sudo, please refer to https://tesseract-ocr.github.io/tessdoc/Installation.html
```

4. Setup the project

Please make sure you have access to the UOE GitLab first.

```
cd ~/
git clone --recurse-submodules git@git.ecdf.ed.ac.uk:s1891075/msc_project.git
pip install -e pixel
cd distro
./install.sh
source ~/.bashrc
cd ../bAbI-tasks
luarocks make babitasks-scm-1.rockspec
cd ..
mkdir -p /exports/eddie/scratch/{UUN}/pixelplus
ln -s /exports/eddie/scratch/{UUN}/pixelplus storage
```

## Prepare Dataset

1. Clone the PIXEL weights, fonts

```
git clone https://huggingface.co/Team-PIXEL/pixel-base storage/pixel-base
```

2. Modify the `HF_HOME` variable to somewhere at `/exports/eddie/scratch/{UUN}/`

3. Run the data preparation scripts

The scripts only collate text files, does not render them into image.

```
python prepare_dataset.py
```
