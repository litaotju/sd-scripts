## Change this for your model's data and model name, and number of steps
export input=./data/mkl.v0.6.2
export model_name="mkl.v0.7.2"
export resume_model="mkl.v0.7.1"
export resume_from="output/${resume_model}/${resume_model}-state"
export steps=10000

######################## almost common across all different models ##########################
REG_DATA_OPTION=""
if [ -d "${input}/class" ]; then
    REG_DATA_OPTION="--reg_data_dir=${input}/class --prior_loss_weight=1.0"
fi

RESUME_OPTION=""
if [ -d ${resume_from} ]; then
    RESUME_OPTION="--resume=${resume_from}"
fi

accelerate launch --num_cpu_threads_per_process 1 \
    train_network.py  \
    --pretrained_model_name_or_path="/home/litao/stable-diffusion-webui/models/Stable-diffusion/uberRealisticPornMerge_urpmv13.safetensors" \
    --mixed_precision=fp16 \
    --train_batch_size=1 \
    --unet_lr=1e-4 \
    --text_encoder_lr=5e-6 \
    --optimizer_type=AdamW8bit \
    --network_module=networks.lora \
    --save_model_as=safetensors  \
    --clip_skip=2 --seed=42 \
    \
    --sample_every_n_epochs 5 \
    \
    --max_train_steps=${steps} \
    --save_every_n_epochs=5 \
    --resolution=256,256 --random_crop --enable_bucket --color_aug \
    \
    --sample_prompts="${input}/prompts.txt" \
    --train_data_dir="${input}/instances" \
    ${REG_DATA_OPTION} \
    --output_name="${model_name}" \
    --output_dir=./output/${model_name} \
    --logging_dir=./output/${model_name} \
    --save_last_n_epochs_state 1 \
    --save_state  \
    --xformers \
    ${RESUME_OPTION}
