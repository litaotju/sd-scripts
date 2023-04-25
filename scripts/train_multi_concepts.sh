#! /usr/bin/bash

set -x

## Change this for your model's data and model name, and number of steps
model_name=$1
epochs=$2
training_data_dir=$3

CHECK_EVERY_N_EPOCHS=5
# optional
resume_from=$4

out_dir=./output/${model_name}-$(date '+%Y-%m-%d-%H-%M-%S')
mkdir -p ${out_dir}
resolution="512,768--random_crop --enable_bucket"
base_model=/home/litao/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors
# base_model=/home/litao/stable-diffusion-webui/models/Stable-diffusion/chilloutmix_NiPrunedFp32Fix.safetensors

if [ ! -e ${base_model} ]; then
    echo "Can not find the base model ${base_model}"
    exit 1
fi
if [ ! -e "${training_data_dir}/prompts.txt" ]; then
    echo "Can not find the sample prompts.txt in directory ${training_data_dir}"
    exit 1
fi

# copy the data to the output dir
mkdir -p ${out_dir}/data/
# "/" here is important, to make sure the content of the directory is copied, not the directory itself
rsync -r "${training_data_dir}/" "${out_dir}/data/"
cp "${training_data_dir}/prompts.txt" ${out_dir}/data/
training_data_dir=${out_dir}/data
sample_prompts=${out_dir}/data/prompts.txt

######################## almost common across all different models ##########################

RESUME_OPTION=""
if [ -e "${resume_from}/pytorch_model.bin" ]; then
    RESUME_OPTION="--resume=${resume_from}"
else
    echo "Can not find the ${resume_from}/pytorch_model.bin, not resume"
fi

negative_prompt="illustration, 3d, sepia, painting, cartoons, sketch, (worst quality:2), (low quality:2), \
(normal quality:2), lowres, bad anatomy, bad hands, normal quality, ((monochrome)), ((grayscale:1.2)), \
futanari, full-package_futanari, penis_from_girl, newhalf, collapsed eyeshadow, multiple eyebrows, \
vaginas in breasts,holes on breasts, fleckles, stretched nipples, nipples on buttocks, analog, analogphoto, signatre, logo,2 faces"

# Train the model
accelerate launch --num_cpu_threads_per_process 1 \
    train_network.py  \
    --pretrained_model_name_or_path=${base_model} \
    --mixed_precision=fp16 \
    --train_batch_size=2 \
    --unet_lr=5e-4 \
    --text_encoder_lr=5e-5 \
    --optimizer_type=AdamW8bit \
    --lr_scheduler=cosine \
    --network_module=networks.lora \
    --save_model_as=safetensors  \
    --clip_skip=2 --seed=42 \
    --network_dim 64 \
    \
    --max_train_epochs=${epochs} \
    --max_token_length=225 \
    --save_every_n_epochs=${CHECK_EVERY_N_EPOCHS} \
    --sample_every_n_epochs=${CHECK_EVERY_N_EPOCHS} \
    --negative_prompt="${negative_prompt}" \
    --save_last_n_epochs_state 1 \
    --save_state  \
    \
    --resolution=${resolution} --color_aug \
    \
    --sample_prompts=${sample_prompts} \
    --train_data_dir=${training_data_dir} \
    --output_name="${model_name}" \
    --output_dir=${out_dir} \
    --logging_dir=${out_dir} \
    --caption_extension ".txt" \
    --xformers \
    ${RESUME_OPTION}

# TODO: do evaluation, and save to the outputs

# Deploy the lora to stable-diffusion-webui
dst=/home/litao/stable-diffusion-webui/models/Lora
if [ ! -e ${dst}/${model_name}.safetensors ]; then
    cp ${out_dir}/${model_name}.safetensors ${dst}/
else 
    echo "Model already exists, not overwriting, please delete the model first."
fi

./scripts/eval_concepts.sh ${out_dir}/${model_name}.safetensors

set +x
