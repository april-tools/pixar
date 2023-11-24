export NUM_GPU_PER_NODE=1
export NUM_NODES=1

torchrun    \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=$NUM_NODES  \
    --nproc-per-node=$NUM_GPU_PER_NODE  \
    train.py \
    --model 'LatentLlama' \
    --exp_type 'debug' \
    --backbone_path storage/checkpoints/debug/lpixel_pretrain/LatentLlama/20231124-020256/20/backbone \
    --dataset_path storage/booksAndWiki2 \
    --shuffle_dataset true \
    --optim 'AdamW' \
    --lr 6e-4 \
    --beta1 0.9 \
    --beta2 0.95 \
    --decay 0.1 \
    --total_steps 3000000 \
    --stop_step 40 \
    --warm_up_step 2000 \
    --save_freq 20 \
    --eval_freq 20 \
    --seed 42 \
    --batch_size 256 \
    --sub_size 8 \
    --dpi 80 \
    --pixels_per_patch 8 \
    --patch_len 2 \
    --num_channel 1 \
    --binary true \
    --rgb false \
    --max_seq_length 720 \
    --mix_precision fp16 \
    --half_coder false \
    --mp_workers 1 \
    --prerendered true \
    --is_continue_train true \
    --num_gpu_per_node $NUM_GPU_PER_NODE \
    --num_node $NUM_NODES  \
