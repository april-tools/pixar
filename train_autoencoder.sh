#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --time=40:30:00

#SBATCH --account=sc118

export NUM_NODES=4
export GPU_PER_NODE=4

source /work/sc118/sc118/yintaotai/.bashrc
conda activate pt2hfpy310

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
    --model 'CNNAutoencoder' \
    --exp_type 'Compressor' \
    --coder_path storage/autoencoders/r4h4c3r1hc64 \
    --render_path storage/pixel-base \
    --dataset_paths storage/enwiki storage/bookcorpus \
    --optim 'AdamW' \
    --lr 1.5e-4 \
    --beta1 0.99 \
    --beta2 0.999 \
    --decay 0.05 \
    --total_steps 10000 \
    --save_freq 500 \
    --best_save_freq 50 \
    --seed 42 \
    --batch_size 256 \
    --sub_size 16 \
    --font_file 'GoNotoCurrent.ttf' \
    --eval_freq 200 \
    --dpi 120 \
    --pixels_per_patch 16 \
    --min_len 900 \
    --max_seq_length 720 \
    --patch_len 6 \
    --mix_precision fp16 \
    --mp_workers 8 \
    --num_gpu_per_node $GPU_PER_NODE \
    --num_node $NUM_NODES  \
