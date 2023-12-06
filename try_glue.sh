export GPU_PER_NODE=1
export NUM_NODES=1


torchrun    \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=$NUM_NODES  \
    --nproc-per-node=$GPU_PER_NODE  \
    train.py \
        --model 'LatentLlamaForSequenceClassification' \
        --exp_type 'bdllama_cola' \
        --backbone_path storage/llama_2_backbone \
        --optim 'AdamW' \
        --finetune_task 'glue' \
        --glue_task 'cola' \
        --lr 3e-5 \
        --beta1 0.9 \
        --beta2 0.95 \
        --decay 0.01 \
        --total_steps 1500 \
        --eval_freq 100 \
        --save_freq 100 \
        --warm_up_step 150 \
        --best_save_freq 150 \
        --seed 42 \
        --batch_size 256 \
        --sub_size 32 \
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


torchrun    \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=$NUM_NODES  \
    --nproc-per-node=$GPU_PER_NODE  \
    train.py \
        --model 'LatentLlamaForSequenceClassification' \
        --exp_type 'bdllama_qqp' \
        --backbone_path storage/llama_2_backbone \
        --optim 'AdamW' \
        --finetune_task 'glue' \
        --glue_task 'qqp' \
        --lr 3e-5 \
        --beta1 0.9 \
        --beta2 0.95 \
        --decay 0.1 \
        --total_steps 15000 \
        --eval_freq 500 \
        --save_freq 500 \
        --warm_up_step 1000 \
        --best_save_freq 1000 \
        --seed 42 \
        --batch_size 256 \
        --sub_size 32 \
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


torchrun    \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=$NUM_NODES  \
    --nproc-per-node=$GPU_PER_NODE  \
    train.py \
        --model 'LatentLlamaForSequenceClassification' \
        --exp_type 'bdllama_qnli' \
        --backbone_path storage/llama_2_backbone \
        --optim 'AdamW' \
        --finetune_task 'glue' \
        --glue_task 'qnli' \
        --lr 3e-5 \
        --beta1 0.9 \
        --beta2 0.95 \
        --decay 0.1 \
        --total_steps 10000 \
        --eval_freq 200 \
        --save_freq 200 \
        --warm_up_step 1000 \
        --best_save_freq 500 \
        --seed 42 \
        --batch_size 256 \
        --sub_size 32 \
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


torchrun    \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=$NUM_NODES  \
    --nproc-per-node=$GPU_PER_NODE  \
    train.py \
        --model 'LatentLlamaForSequenceClassification' \
        --exp_type 'bdllama_rte' \
        --backbone_path storage/llama_2_backbone \
        --optim 'AdamW' \
        --finetune_task 'glue' \
        --glue_task 'rte' \
        --lr 3e-5 \
        --beta1 0.9 \
        --beta2 0.95 \
        --decay 0.01 \
        --total_steps 2000 \
        --eval_freq 50 \
        --save_freq 50 \
        --warm_up_step 100 \
        --best_save_freq 100 \
        --seed 42 \
        --batch_size 64 \
        --sub_size 32 \
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


torchrun    \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=$NUM_NODES  \
    --nproc-per-node=$GPU_PER_NODE  \
    train.py \
        --model 'LatentLlamaForSequenceClassification' \
        --exp_type 'bdllama_stsb' \
        --backbone_path storage/llama_2_backbone \
        --optim 'AdamW' \
        --finetune_task 'glue' \
        --glue_task 'stsb' \
        --lr 3e-5 \
        --beta1 0.9 \
        --beta2 0.95 \
        --decay 0.01 \
        --total_steps 15000 \
        --eval_freq 500 \
        --save_freq 500 \
        --warm_up_step 1000 \
        --best_save_freq 200 \
        --seed 42 \
        --batch_size 64 \
        --sub_size 32 \
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
