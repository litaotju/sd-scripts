accelerate launch --num_cpu_threads_per_process 1 \
    train_network.py  \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --xformers --mixed_precision=fp16 \
    --train_batch_size=1 \
    --learning_rate=1e-4  \
    --optimizer_type=AdamW8bit \
    --network_module=networks.lora \
    --save_model_as=safetensors  \
    --clip_skip=2 --seed=42 \
    \
    --sample_every_n_epochs 1 \
    \
    --max_train_steps=1000 \
    --save_every_n_epochs=5 \
    --resolution=256,256 --random_crop --enable_bucket --color_aug \
    --sample_prompts="data/mila-face/prompts.txt" \
    --train_data_dir="./data/mila-face/instances" \
    --output_name="milaface" \
    --output_dir=./output/milaface

    #--reg_data_dir=./data/mila-face/class --prior_loss_weight=1.0  \