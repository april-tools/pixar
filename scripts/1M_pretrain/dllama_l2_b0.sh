export NUM_GPU_PER_NODE=4
export NUM_NODES=1

export OMP_NUM_THREADS=64


torchrun    \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=$NUM_NODES  \
    --nproc-per-node=$NUM_GPU_PER_NODE  \
    train.py \
    --model 'LatentLlama' \
    --exp_type 'pretrain' \
    --backbone_path /exports/eddie3_homes_local/s1891075/msc_project/storage/checkpoints/pretrain/lpixel_pretrain/LatentLlama/20231201-210422/860000/backbone \
    --dataset_path /home/s1891075/msc_project/storage/BooksAndWiki2 \
    --shuffle_dataset false \
    --optim 'AdamW' \
    --lr 3e-4 \
    --beta1 0.9 \
    --beta2 0.95 \
    --decay 0.1 \
    --total_steps 900000 \
    --stop_step 900000 \
    --warm_up_step 2000 \
    --save_freq 10000 \
    --eval_freq 10000 \
    --seed 42 \
    --batch_size 384 \
    --sub_size 96 \
    --dpi 80 \
    --font_size 8 \
    --font_file PixeloidSans-mLxMm.ttf \
    --pixels_per_patch 8 \
    --patch_len 2 \
    --num_channel 1 \
    --binary true \
    --rgb false \
    --max_seq_length 720 \
    --mix_precision fp16 \
    --half_coder false \
    --mp_workers 16 \
    --prerendered false \
    --is_continue_train true \
    --num_gpu_per_node $NUM_GPU_PER_NODE \
    --num_node $NUM_NODES  \
