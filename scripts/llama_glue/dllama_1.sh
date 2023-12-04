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

export BACKBONE_PATH='storage/shared/checkpoints/dllama_pretrain/lpixel_pretrain/LatentLlama/20231118-205922/100000/backbone'
export BATCH_SIZE=8
export MAX_NUM_PATCH=360
export PATCH_LEN=1

export EXP_NAME='dllama_1'

srun torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$GPU_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-id=123 \
    --rdzv-endpoint=$head_node_ip \
    train.py --model LatentLlamaForSequenceClassification \
        --exp_type ${EXP_NAME}_mrpc_ \
        --backbone_path $BACKBONE_PATH \
        --checkpoint_path /work/sc118/sc118/shared/checkpoints \
        --optim AdamW \
        --finetune_task glue \
        --glue_task mrpc \
        --lr 3e-5 --beta1 0.9 --beta2 0.95 --decay 0.01 --total_steps 100 --eval_freq 20 --save_freq 20 --warm_up_step 10 --best_save_freq 50 --seed 42 --batch_size 192 \
        --sub_size 6 \
        --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH \
        --num_channel 1 --binary true --rgb false --mix_precision fp16 --half_coder true --mp_workers 8 \
        --num_gpu_per_node $GPU_PER_NODE \
        --num_node $NUM_NODES

srun torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$GPU_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-id=123 \
    --rdzv-endpoint=$head_node_ip \
    train.py --model LatentLlamaForSequenceClassification \
        --exp_type ${EXP_NAME}_wnli_ \
        --backbone_path $BACKBONE_PATH \
        --optim AdamW --finetune_task glue --glue_task wnli --lr 3e-5 --beta1 0.9 --beta2 0.95 --decay 0.01 --total_steps 200 --eval_freq 5 --save_freq 5 --warm_up_step 20 --best_save_freq 10 --seed 42 --batch_size 64 \
        --sub_size 4 --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary true --rgb false --mix_precision fp16 --half_coder true --mp_workers 8 \
        --num_gpu_per_node $GPU_PER_NODE \
        --num_node $NUM_NODES

srun torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$GPU_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-id=123 \
    --rdzv-endpoint=$head_node_ip \
    train.py --model LatentLlamaForSequenceClassification \
        --exp_type ${EXP_NAME}_mnli_ \
        --backbone_path $BACKBONE_PATH \
        --optim AdamW --finetune_task glue --glue_task mnli --lr 3e-5 --beta1 0.9 --beta2 0.95 --decay 0.1 --total_steps 8000 --eval_freq 500 --save_freq 500 --warm_up_step 1000 --best_save_freq 1000 --seed 42 --batch_size 256 \
        --sub_size $BATCH_SIZE --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary true --rgb false --mix_precision fp16 --half_coder true --mp_workers 8 \
        --num_gpu_per_node $GPU_PER_NODE \
        --num_node $NUM_NODES

srun torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$GPU_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-id=123 \
    --rdzv-endpoint=$head_node_ip \
    train.py --model LatentLlamaForSequenceClassification \
        --exp_type ${EXP_NAME}_sst2_ \
        --backbone_path $BACKBONE_PATH \
        --optim AdamW --finetune_task glue --glue_task sst2 --lr 3e-5 --beta1 0.9 --beta2 0.95 --decay 0.01 --total_steps 2000 --eval_freq 200 --save_freq 200 --warm_up_step 200 --best_save_freq 300 --seed 42 --batch_size 256 \
        --sub_size $BATCH_SIZE --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary true --rgb false --mix_precision fp16 --half_coder true --mp_workers 8 \
        --num_gpu_per_node $GPU_PER_NODE \
        --num_node $NUM_NODES

srun torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$GPU_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-id=123 \
    --rdzv-endpoint=$head_node_ip \
    train.py --model LatentLlamaForSequenceClassification \
        --exp_type ${EXP_NAME}_stsb_ \
        --backbone_path $BACKBONE_PATH \
        --optim AdamW --finetune_task glue --glue_task stsb --lr 3e-5 --beta1 0.9 --beta2 0.95 --decay 0.01 --total_steps 15000 --eval_freq 500 --save_freq 500 --warm_up_step 1000 --best_save_freq 200 --seed 42 --batch_size 64 \
        --sub_size 4 --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary true --rgb false --mix_precision fp16 --half_coder true --mp_workers 8 \
        --num_gpu_per_node $GPU_PER_NODE \
        --num_node $NUM_NODES

srun torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$GPU_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-id=123 \
    --rdzv-endpoint=$head_node_ip \
    train.py --model LatentLlamaForSequenceClassification \
        --exp_type ${EXP_NAME}_rte_ \
        --backbone_path $BACKBONE_PATH \
        --optim AdamW --finetune_task glue --glue_task rte --lr 3e-5 --beta1 0.9 --beta2 0.95 --decay 0.01 --total_steps 500 --eval_freq 50 --save_freq 50 --warm_up_step 50 --best_save_freq 100 --seed 42 --batch_size 64 \
        --sub_size 4 --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary true --rgb false --mix_precision fp16 --half_coder true --mp_workers 8 \
        --num_gpu_per_node $GPU_PER_NODE \
        --num_node $NUM_NODES

srun torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$GPU_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-id=123 \
    --rdzv-endpoint=$head_node_ip \
    train.py --model LatentLlamaForSequenceClassification \
        --exp_type ${EXP_NAME}_qnli_ \
        --backbone_path $BACKBONE_PATH \
        --optim AdamW --finetune_task glue --glue_task qnli --lr 3e-5 --beta1 0.9 --beta2 0.95 --decay 0.1 --total_steps 4000 --eval_freq 200 --save_freq 200 --warm_up_step 500 --best_save_freq 500 --seed 42 --batch_size 256 \
        --sub_size $BATCH_SIZE --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary true --rgb false --mix_precision fp16 --half_coder true --mp_workers 8 \
        --num_gpu_per_node $GPU_PER_NODE \
        --num_node $NUM_NODES
    
srun torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$GPU_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-id=123 \
    --rdzv-endpoint=$head_node_ip \
    train.py --model LatentLlamaForSequenceClassification \
        --exp_type ${EXP_NAME}_qqp_ \
        --backbone_path $BACKBONE_PATH \
        --optim AdamW --finetune_task glue --glue_task qqp --lr 3e-5 --beta1 0.9 --beta2 0.95 --decay 0.1 --total_steps 8000 --eval_freq 500 --save_freq 500 --warm_up_step 1000 --best_save_freq 1000 --seed 42 --batch_size 256 \
        --sub_size $BATCH_SIZE --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary true --rgb false --mix_precision fp16 --half_coder true --mp_workers 8 \
        --num_gpu_per_node $GPU_PER_NODE \
        --num_node $NUM_NODES

srun torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$GPU_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-id=123 \
    --rdzv-endpoint=$head_node_ip \
    train.py --model LatentLlamaForSequenceClassification \
        --exp_type ${EXP_NAME}_cola_ \
        --backbone_path $BACKBONE_PATH \
        --optim AdamW --finetune_task glue --glue_task cola --lr 3e-5 --beta1 0.9 --beta2 0.95 --decay 0.01 --total_steps 500 --eval_freq 100 --save_freq 100 --warm_up_step 50 --best_save_freq 150 --seed 42 --batch_size 256 \
        --sub_size $BATCH_SIZE --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary true --rgb false --mix_precision fp16 --half_coder true --mp_workers 8 \
        --num_gpu_per_node $GPU_PER_NODE \
        --num_node $NUM_NODES
