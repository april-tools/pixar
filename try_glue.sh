export GPU_PER_NODE=1
export NUM_NODES=1

torchrun    \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=$NUM_NODES  \
    --nproc-per-node=$GPU_PER_NODE  \
    train.py \
        --model 'LatentLlamaForSequenceClassification' \
        --exp_type 'bdllama_sst2' \
        --backbone_path storage/llama_2_backbone \
        --optim 'AdamW' \
        --finetune_task 'glue' \
        --glue_task 'sst2' \
        --lr 3e-5 \
        --beta1 0.9 \
        --beta2 0.95 \
        --decay 0.01 \
        --total_steps 2000 \
        --eval_freq 200 \
        --save_freq 200 \
        --warm_up_step 200 \
        --best_save_freq 300 \
        --seed 42 \
        --batch_size 256 \
        --sub_size 8 \
        --font_file 'PixeloidSans-mLxMm.ttf' \
        --dpi 80 \
        --pixels_per_patch 8 \
        --patch_len 2 \
        --max_seq_length 720 \
        --num_channel 1 \
        --binary true \
        --rgb false \
        --mix_precision fp16 \
        --half_coder true \
        --mp_workers 8 \
        --num_gpu_per_node $GPU_PER_NODE \
        --num_node $NUM_NODES  \
