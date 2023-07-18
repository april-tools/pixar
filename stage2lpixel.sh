#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --time=40:00:00
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
    --exp_type 'lpixel_pretrain2' \
    --backbone_path storage/checkpoints/lpixel_pretrain1/lpixel_pretrain/LPixelForPreTraining/20230718-005016/1500/backbone \
    --coder_path storage/SD2_VQGAN \
    --render_path storage/pixel-base \
    --dataset_paths storage/enwiki storage/bookcorpus \
    --lr 1e-4 \
    --beta1 0.99 \
    --beta2 0.999 \
    --decay 0.05 \
    --stage 2 \
    --total_steps 100000 \
    --save_freq 1000 \
    --best_save_freq 100 \
    --seed 42 \
    --batch_size 240 \
    --sub_size 5 \
    --dpi 240 \
    --pixels_per_patch 32 \
    --mix_precision fp16 \
    --half_coder true \
    --mp_workers 8 \
    --num_gpu_per_node $GPU_PER_NODE \
    --num_node $NUM_NODES  \
