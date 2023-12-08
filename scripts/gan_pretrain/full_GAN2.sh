export NUM_GPU_PER_NODE=2
export NUM_NODES=1

export OMP_NUM_THREADS=32

torchrun    \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=$NUM_NODES  \
    --nproc-per-node=$NUM_GPU_PER_NODE  \
    train.py \
    --task 'gan_pretrain' \
    --model 'LatentLlama' \
    --exp_type 'full_gan' \
    --backbone_path /home/s1891075/msc_project/storage/checkpoints/pretrain/lpixel_pretrain/LatentLlama/20231202-215356/900000/backbone \
    --dataset_path storage/BooksAndWiki2 \
    --discriminator_path self \
    --shuffle_dataset false \
    --optim 'AdamW' \
    --lr 3e-5 \
    --beta1 0.9 \
    --beta2 0.95 \
    --decay 0.1 \
    --total_steps 5000 \
    --stop_step 5000 \
    --warm_up_step 100 \
    --save_freq 200 \
    --eval_freq 200 \
    --gan_lr 3e-5 \
    --gan_lr_warm_up_steps 100 \
    --gan_total_steps 5000 \
    --gan_ratio 7.0 \
    --gan_ratio_warm_up_steps 100 \
    --seed 42 \
    --batch_size 128 \
    --sub_size 64 \
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

torchrun    \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=$NUM_NODES  \
    --nproc-per-node=$NUM_GPU_PER_NODE  \
    train.py \
    --task 'gan_pretrain' \
    --model 'LatentLlama' \
    --exp_type 'full_gan' \
    --backbone_path /home/s1891075/msc_project/storage/checkpoints/pretrain/lpixel_pretrain/LatentLlama/20231202-215356/900000/backbone \
    --dataset_path storage/BooksAndWiki2 \
    --discriminator_path self \
    --shuffle_dataset false \
    --optim 'AdamW' \
    --lr 3e-5 \
    --beta1 0.9 \
    --beta2 0.95 \
    --decay 0.1 \
    --total_steps 5000 \
    --stop_step 5000 \
    --warm_up_step 100 \
    --save_freq 200 \
    --eval_freq 200 \
    --gan_lr 3e-5 \
    --gan_lr_warm_up_steps 100 \
    --gan_total_steps 5000 \
    --gan_ratio 1.0 \
    --gan_ratio_warm_up_steps 100 \
    --seed 42 \
    --batch_size 128 \
    --sub_size 64 \
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
