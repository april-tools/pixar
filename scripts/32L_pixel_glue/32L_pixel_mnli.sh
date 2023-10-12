#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --time=70:30:0
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
    --model 'LPixelForClassification' \
    --exp_type '32L_pixel_glue_mnli' \
    --backbone_path storage/checkpoints/lpixel_pretrain2/lpixel_pretrain/LPixelForPreTraining/20230718-020315/99000/backbone \
    --coder_path storage/SD2_VQGAN \
    --render_path storage/pixel-base \
    --optim 'AdamW' \
    --finetune_task 'glue' \
    --glue_task 'mnli' \
    --lr 3e-5 \
    --beta1 0.9 \
    --beta2 0.999 \
    --decay 0.00 \
    --stage 2 \
    --total_steps 30000 \
    --eval_freq 200 \
    --begin_eval 5000 \
    --save_freq 1000 \
    --warm_up_step 100 \
    --best_save_freq 50 \
    --latent_norm False \
    --seed 42 \
    --batch_size 192 \
    --sub_size 6 \
    --font_file 'GoNotoCurrent.ttf' \
    --dpi 240 \
    --pixels_per_patch 32 \
    --image_size 3 32 16928 \
    --latent_size 4 4 2116 \
    --mix_precision fp16 \
    --half_coder true \
    --mp_workers 4 \
    --num_gpu_per_node $GPU_PER_NODE \
    --num_node $NUM_NODES  \
