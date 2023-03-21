#! /usr/bin/bash

set -x

## Change this for your model's data and model name, and number of steps
person=$1
training_data_dir=$2
epochs=$3

## 
out_dir=./output/${person}-$(date '+%Y-%m-%d-%H-%M-%S')
mkdir -p ${out_dir}
resolution="512,512 --random_crop --enable_bucket"
base_model=/home/litao/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors

#####Set up the model name and sample prompts
model_name=${person}-lora
resume_from="/tmp/xxxxxx"
sample_prompts=${out_dir}/prompts.txt

echo "a woman ${person}
${person} a woman with long hair wearing a pink top
a woman ${person}, headshot photo with sun glass
a photo of a woman ${person}, nude white female headshot with pink bra
${person} standing by a table, holding an apple, with green hat
a woman ${person} in a black lingerie posing for a picture in a mirror with her hands on her breasts
" > ${sample_prompts}

# copy the data to the output dir
mkdir -p ${out_dir}/data/
cp -r "${training_data_dir}/" "${out_dir}/data/"
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
    --train_batch_size=4 \
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
    --xformers \
    ${RESUME_OPTION}

# Do evaluation, and save to the outputs

# TODO: deploy the lora to stable-diffusion-webui

set +x