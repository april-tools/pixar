export NUM_GPU_PER_NODE=1
export NUM_NODES=1

# export OMP_NUM_THREADS=64


torchrun    \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=$NUM_NODES  \
    --nproc-per-node=$NUM_GPU_PER_NODE  \
    train.py \
    --task 'gan_pretrain' \
    --model 'LatentLlama' \
    --exp_type 'gan_pretrain' \
    --backbone_path /home/tai/src/projects/msc_project/storage/llama_2_backbone \
    --dataset_path storage/booksAndWiki2 \
    --discriminator_path /home/tai/src/projects/msc_project/storage/discriminator/l2b \
    --shuffle_dataset false \
    --optim 'AdamW' \
    --lr 1e-4 \
    --beta1 0.9 \
    --beta2 0.95 \
    --decay 0.1 \
    --total_steps 900000 \
    --stop_step 900000 \
    --warm_up_step 2000 \
    --save_freq 200 \
    --eval_freq 200 \
    --gan_lr 3e-4 \
    --gan_lr_warm_up_steps 100 \
    --gan_total_steps 500000 \
    --gan_ratio 0.1 \
    --gan_ratio_warm_up_steps 1000 \
    --seed 42 \
    --batch_size 128 \
    --sub_size 32 \
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
    --is_continue_train false \
    --num_gpu_per_node $NUM_GPU_PER_NODE \
    --num_node $NUM_NODES  \
