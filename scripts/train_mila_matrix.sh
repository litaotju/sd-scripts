#! /usr/bin/bash

for dim in 8 16 32; do
for lr in 1e-5 1e-4; do
    echo "Training with dim=$dim and lr=$lr"
    OUT=/mnt/mass01/lora-training/output/mila-dim-${dim}-lr-${lr}
    mkdir ${OUT}
    accelerate launch --num_cpu_threads_per_process 1 \
       train_network.py  \
       --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
       --xformers --mixed_precision=fp16 \
       --train_batch_size=1 \
       --learning_rate=${lr}  \
       --optimizer_type=AdamW8bit \
       --network_module=networks.lora \
       --save_model_as=safetensors  \
       --clip_skip=2 --seed=42 \
       \
       --sample_every_n_epochs 1 \
       \
       --network_dim ${dim}\
       --max_train_steps=3000 \
       --save_every_n_epochs=5 \
       --resolution=400,600 --random_crop --enable_bucket --color_aug \
       --sample_prompts="data/mila/prompts.txt" \
       --train_data_dir="./data/mila/instances" \
       --reg_data_dir=./data/mila/class --prior_loss_weight=1.0  \
       --output_name="mila-dim-${dim}-lr-${lr}" \
       --output_dir=${OUT} |& tee ${OUT}/log.txt
    sleep 60s
done
done