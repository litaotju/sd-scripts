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
# base_model=/home/litao/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors
base_model=/home/litao/stable-diffusion-webui/models/Stable-diffusion/chilloutmix_NiPrunedFp32Fix.safetensors

#####Set up the model name and sample prompts
resume_from="/tmp/xxxxxxxx"
sample_prompts=${out_dir}/prompts.txt

style="(8k, RAW photo, best quality, masterpiece:1.2), (realistic:1.3), (photorealistic:1.3), ultra-detailed, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3, best quality, beautiful lighting --n illustration, 3d, sepia, painting, cartoons, sketch, (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad hands, normal quality, ((monochrome)), ((grayscale:1.2)), futanari, full-package_futanari, penis_from_girl, newhalf, collapsed eyeshadow, multiple eyebrows, vaginas in breasts,holes on breasts, fleckles, stretched nipples, nipples on buttocks, analog, analogphoto, signatre, logo,2 faces"
echo "
a woman is riding a big cock on a couch --h 768 --w 512
a woman is riding a big cock ${style} --h 768 --w 512
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
    --unet_lr=4e-4 \
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
