## 1. Environment Setup

### 1. Create the environment

Text images are rendered differently under different versions of cairo and fonttools. I higly recommand to create the environment step by step using the following commands.

```
conda create -n {name} python=3.10
conda activate {name}
conda install -c conda-forge pycairo=1.24.0 cairo=1.16.0 fonttools=4.25.0 pygobject=3.42.1 manimpango=0.4.1 # make sure the version matches, othewise the rendered text may different
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.34.1 datasets==2.14.6 tokenizers==0.14.1 diffusers["torch"]==0.17.1
pip install opencv-python==4.6.0.66 opencv-contrib-python==4.6.0.66
pip install ipywidgets wandb==0.15.12 pytesseract==0.3.10 editdistance==0.6.2 adamr scikit-image==0.22.0 scikit-learn==1.3.0
pip install nltk==3.8.1 apache-beam==2.51.0
pip install paddlepaddle==2.5.2 paddleocr==2.7.0.3
pip install flash-attn --no-build-isolation # install FlashAttention https://github.com/Dao-AILab/flash-attention
sudo apt install tesseract-ocr # if you cannot use sudo, please refer to https://tesseract-ocr.github.io/tessdoc/Installation.html
```

2. Setup the project directory

Here are commands to setup the project folder.

```
cd {where you want to put the project folder}
git clone --recurse-submodules git@github.com:april-tools/pixar.git
pip install -e pixel
cd distro
./install.sh
source ~/.bashrc
cd ../bAbI-tasks
luarocks make babitasks-scm-1.rockspec
cd ..
mkdir storage
```

### 2. Prepare the pretraining dataset

1. Clone the PIXEL weights, fonts

```
git clone https://huggingface.co/Team-PIXEL/pixel-base {where you want to put the model}
```

3. Run the data preparation scripts

The scripts only collate text files, does not render them into image. By default, we render images during the training.

```
python prepare_dataset.py
```

## 2. Pretraining

When you run the scripts, remember to modify hyperparameters to suit your environment or change the model settings. A detailed hyperparameter instruction is coming soon.

1. Initialize the backbone model

```
python scripts/init_models.py
```

2. Stage one pretraining


```
bash scripts/1M_pretrain/dllama_l2_b0.sh
```

2. Stage two pretraining

```
bash scripts/gan_pretrain/full_GAN1.sh
```

## 3. Download the pretrained checkpoint

Here is the model checkpoint used to test the Lambda and bAbI performance in the paper. It was trained with 1M stage 1 and 200 stage 2 steps.

[https://drive.google.com/file/d/1uFr78VttIfOOqYxyCcsoc7ewsQvU_BVf/view?usp=sharing
](https://drive.google.com/file/d/1ngfKBmCL_nEa2om9ifJHOaf4SDKQ-eMP/view?usp=drive_link)

Here are some samples generated from it: 

https://drive.google.com/drive/folders/1vHmF0UHGKAzhtOUtP55_m6VrfVQIHc4K?usp=drive_link

## 4. Run the downstream tasks

1. GLUE

```
bash scripts/llama_glue/dllama_2b1M.sh
```

2. bAbI

You can use babi.ipynb to evalueate the performance on the bAbI benchmark.


3. LAMBADA

You can use lambada.ipynb to evalueate the performance on the bAbI benchmark.


