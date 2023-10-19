#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --time=40:30:0
#SBATCH --qos=gpu
#SBATCH --exclusive

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
    --model 'LatentGPT2' \
    --exp_type 'debug' \
    --backbone_path storage/checkpoints/24D_gpt2_1/lpixel_pretrain/LatentGPT2/20230810-192053/2000/backbone \
    --compressor_name CNNAutoencoder \
    --dataset_paths storage/enwiki storage/bookcorpus \
    --optim 'AdamW' \
    --lr 1.5e-4 \
    --beta1 0.99 \
    --beta2 0.999 \
    --decay 0.05 \
    --total_steps 100 \
    --save_freq 20 \
    --best_save_freq 50 \
    --seed 42 \
    --batch_size 256 \
    --sub_size 8 \
    --font_file PixeloidSans-mLxMm.ttf \
    --dpi 80 \
    --pixels_per_patch 8 \
    --patch_len 5 \
    --num_channel 1 \
    --binary true \
    --rgb false \
    --max_seq_length 2000 \
    --min_len 800 \
    --mix_precision fp16 \
    --half_coder false \
    --mp_workers 8 \
    --num_gpu_per_node $GPU_PER_NODE \
    --num_node $NUM_NODES  \
