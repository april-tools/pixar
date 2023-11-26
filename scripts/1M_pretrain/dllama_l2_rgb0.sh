#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
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

export OMP_NUM_THREADS=32

srun torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$GPU_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-id=123 \
    --rdzv-endpoint=$head_node_ip \
    train.py \
    --model 'LatentLlama' \
    --exp_type 'pretrain' \
    --backbone_path /work/sc118/sc118/shared/checkpoints/100kModels/dllama_2_rgb_100k/backbone \
    --dataset_path /work/sc118/sc118/shared/BooksAndWiki2 \
    --shuffle_dataset false \
    --optim 'AdamW' \
    --lr 3e-4 \
    --beta1 0.9 \
    --beta2 0.95 \
    --decay 0.1 \
    --total_steps 1000000 \
    --stop_step 1000000 \
    --warm_up_step 2000 \
    --save_freq 10000 \
    --eval_freq 10000 \
    --seed 42 \
    --batch_size 384 \
    --sub_size 24 \
    --dpi 80 \
    --font_size 8 \
    --font_file PixeloidSans-mLxMm.ttf \
    --pixels_per_patch 8 \
    --patch_len 2 \
    --num_channel 3 \
    --binary false \
    --rgb true \
    --max_seq_length 720 \
    --mix_precision fp16 \
    --half_coder false \
    --mp_workers 8 \
    --prerendered false \
    --is_continue_train true \
    --num_gpu_per_node $GPU_PER_NODE \
    --num_node $NUM_NODES  \
