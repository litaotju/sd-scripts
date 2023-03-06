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
    --max_train_steps=500 \
    --save_every_n_epochs=5 \
    --resolution=512,512 --random_crop --enable_bucket --color_aug \
    --sample_prompts="a photo of a sks dog in the bucket" \
    --train_data_dir="./data/sks/instances" \
    --reg_data_dir=./data/sks/class --prior_loss_weight=1.0  \
    --output_name="sks" \
    --output_dir=./output/sks
