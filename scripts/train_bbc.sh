#! /usr/bin/bash

set -x

## Change this for your model's data and model name, and number of steps
model_name=$1
epochs=$2
training_data_dir=$3

## 
out_dir=./output/${model_name}-$(date '+%Y-%m-%d-%H-%M-%S')
mkdir -p ${out_dir}
resolution="512,768 --random_crop --enable_bucket"
base_model=/home/litao/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors

#####Set up the model name and sample prompts
resume_from="/tmp/xxxxxx"
sample_prompts=${out_dir}/prompts.txt

echo "
a woman is sucking a big black cock --h 768 --w 512
a woman is licking a big black cock --h 768 --w 512
a woman is blowing a big black cock kneeing --h 768 --w 512
a woman is riding a big black cock in pussy, with long hair wearing a white bra --h 768 --w 512
a woman is riding a big black cock in anal, with long hair wearing a white bra --h 768 --w 512
a woman is fucked by big block cock m-legs pov --h 768 --w 512
" > ${sample_prompts}

# copy the data to the output dir
mkdir -p ${out_dir}/data/
rsync -r "${training_data_dir}/" "${out_dir}/data/"
training_data_dir=${out_dir}/data

######################## almost common across all different models ##########################
REG_DATA_OPTION=""

RESUME_OPTION=""
if [ -d ${resume_from} ]; then
    RESUME_OPTION="--resume=${resume_from}"
fi

# Train the model
accelerate launch --num_cpu_threads_per_process 1 \
    train_network.py  \
    --pretrained_model_name_or_path=${base_model} \
    --mixed_precision=fp16 \
    --train_batch_size=2 \
    --unet_lr=2e-4 \
    --text_encoder_lr=5e-5 \
    --optimizer_type=AdamW8bit \
    --lr_scheduler=cosine \
    --network_module=networks.lora \
    --save_model_as=safetensors  \
    --clip_skip=2 --seed=42 \
    --network_dim 64 \
    \
    --max_train_epochs=${epochs} \
    --save_every_n_epochs=5 \
    --sample_every_n_epochs=5 \
    --save_last_n_epochs_state 1 \
    --save_state  \
    \
    --resolution=${resolution} --color_aug \
    \
    --sample_prompts=${sample_prompts} \
    --train_data_dir=${training_data_dir} \
    ${REG_DATA_OPTION} \
    --output_name="${model_name}" \
    --output_dir=${out_dir} \
    --logging_dir=${out_dir} \
    --caption_extension ".txt" \
    --xformers \
    ${RESUME_OPTION}

# Do evaluation, and save to the outputs

# TODO: deploy the lora to stable-diffusion-webui

set +x
