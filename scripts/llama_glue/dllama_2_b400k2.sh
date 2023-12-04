#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --time=10:30:0
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

export BACKBONE_PATH='/work/sc118/sc118/yintaotai/msc_project/storage/shared/checkpoints/400kModels/dllama_2_b/backbone'
export COMPRESSOR_PATH=''
export COMPRESSOR_NAME=''
export BATCH_SIZE=8
export MAX_NUM_PATCH=720
export PATCH_LEN=2
export WANDB__SERVICE_WAIT=300
export BINARY=true
export RGB=false

export EXP_NAME='dllama_2'


srun torchrun \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$GPU_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-id=123 \
    --rdzv-endpoint=$head_node_ip \
    train.py --model LatentLlamaForSequenceClassification \
        --exp_type ${EXP_NAME}_wnli_ \
        --backbone_path $BACKBONE_PATH \
        --optim AdamW --finetune_task glue --glue_task wnli --lr 3e-5 --beta1 0.9 --beta2 0.95 --decay 0.01 --total_steps 20 --eval_freq 1 --save_freq 5 --warm_up_step 2 --best_save_freq 10 --seed 42 --batch_size 128 \
        --sub_size 8 --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary $BINARY --rgb $RGB --mix_precision fp16 --half_coder true --mp_workers 8 \
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
        --optim AdamW --finetune_task glue --glue_task wnli --lr 1e-5 --beta1 0.9 --beta2 0.95 --decay 0.01 --total_steps 20 --eval_freq 3 --save_freq 5 --warm_up_step 2 --best_save_freq 10 --seed 42 --batch_size 128 \
        --sub_size 8 --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary $BINARY --rgb $RGB --mix_precision fp16 --half_coder true --mp_workers 8 \
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
        --optim AdamW --finetune_task glue --glue_task stsb --lr 1e-5 --beta1 0.9 --beta2 0.95 --decay 0.01 --total_steps 1000 --eval_freq 100 --save_freq 100 --warm_up_step 100 --best_save_freq 200 --seed 42 --batch_size 64 \
        --sub_size 4 --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary $BINARY --rgb $RGB --mix_precision fp16 --half_coder true --mp_workers 8 \
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
        --optim AdamW --finetune_task glue --glue_task stsb --lr 3e-5 --beta1 0.9 --beta2 0.95 --decay 0.01 --total_steps 1000 --eval_freq 100 --save_freq 100 --warm_up_step 100 --best_save_freq 200 --seed 42 --batch_size 64 \
        --sub_size 4 --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary $BINARY --rgb $RGB --mix_precision fp16 --half_coder true --mp_workers 8 \
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
        --optim AdamW --finetune_task glue --glue_task stsb --lr 1e-5 --beta1 0.9 --beta2 0.95 --decay 0.01 --total_steps 2000 --eval_freq 100 --save_freq 100 --warm_up_step 100 --best_save_freq 200 --seed 42 --batch_size 64 \
        --sub_size 4 --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary $BINARY --rgb $RGB --mix_precision fp16 --half_coder true --mp_workers 8 \
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
        --optim AdamW --finetune_task glue --glue_task stsb --lr 6e-5 --beta1 0.9 --beta2 0.95 --decay 0.01 --total_steps 2000 --eval_freq 100 --save_freq 100 --warm_up_step 100 --best_save_freq 200 --seed 42 --batch_size 64 \
        --sub_size 4 --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary $BINARY --rgb $RGB --mix_precision fp16 --half_coder true --mp_workers 8 \
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
        --optim AdamW --finetune_task glue --glue_task stsb --lr 3e-5 --beta1 0.9 --beta2 0.95 --decay 0.01 --total_steps 2000 --eval_freq 100 --save_freq 100 --warm_up_step 100 --best_save_freq 200 --seed 42 --batch_size 64 \
        --sub_size 4 --font_file PixeloidSans-mLxMm.ttf --dpi 80 --pixels_per_patch 8 \
        --patch_len $PATCH_LEN \
        --max_seq_length $MAX_NUM_PATCH --num_channel 1 --binary $BINARY --rgb $RGB --mix_precision fp16 --half_coder true --mp_workers 8 \
        --num_gpu_per_node $GPU_PER_NODE \
        --num_node $NUM_NODES


