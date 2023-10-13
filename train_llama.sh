#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --time=5:30:0
#SBATCH --qos=gpu
#SBATCH --exclusive

export NUM_NODES=4
export GPU_PER_NODE=4

#source /work/sc118/sc118/yintaotai/.bashrc
#conda activate pt2hfpy310
#activate the base  = source .bashrc
__conda_setup="$('/work/sc118/sc118/xliao11/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/work/sc118/sc118/xliao11/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/work/sc118/sc118/xliao11/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/work/sc118/sc118/xliao11/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate latent

module load nvidia/nvhpc/22.11

echo $SLURM_JOB_NODELIST

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Head node IP: $head_node_ip

export OMP_NUM_THREADS=20

srun torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$GPU_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-id=123 \
    --rdzv-endpoint=$head_node_ip \
    train.py \
    --model 'LatentGPT2' \
    --exp_type '24D_gpt2_1' \
    --backbone_path storage/gpt2 \
    --render_path storage/pixel-base \
    --dataset_paths storage/enwiki storage/bookcorpus \
    --optim 'AdamW' \
    --lr 1.5e-4 \
    --beta1 0.99 \
    --beta2 0.999 \
    --decay 0.05 \
    --stage 1 \
    --total_steps 2000 \
    --save_freq 500 \
    --best_save_freq 50 \
    --seed 42 \
    --batch_size 192 \
    --sub_size 6 \
    --font_file 'GoNotoCurrent.ttf' \
    --dpi 180 \
    --pixels_per_patch 24 \
    --image_size 3 24 12696 \
    --latent_size 3 24 12696 \
    --mix_precision fp16 \
    --half_coder true \
    --mp_workers 8 \
    --num_gpu_per_node $GPU_PER_NODE \
    --num_node $NUM_NODES  \

