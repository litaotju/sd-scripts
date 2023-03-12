## Change this for your model's data and model name, and number of steps
export input=./data/mkl.v0.8
export model_name="mkl.v0.8.1"
export resume_model="mkl.v0.8.0"
export resume_from="output/${resume_model}/${resume_model}-state"
export steps=20000
export prompt=gen.txt

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
    --text_encoder_lr=1e-5 \
    --optimizer_type=AdamW8bit \
    --network_module=networks.lora \
    --save_model_as=safetensors  \
    --clip_skip=2 --seed=42 \
    \
    --sample_every_n_epochs 1 \
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

cp ./output/${model_name}/${model_name}.safetensors   ~/stable-diffusion-webui/models/Lora

for strength in 0.0 0.2 0.4 0.6 0.8 1.0; do
    python ./gen_img_diffusers.py --from_file $prompt  --images_per_prompt 1 \
    --outdir output/${model_name}/generated/ --fp16 --xformers --batch_size 6 \
    --network_module networks.lora --network_weights output/${model_name}/${model_name}${epoch}.safetensors  --network_mul ${strength} \
    --ckpt ~/stable-diffusion-webui/models/Stable-diffusion/uberRealisticPornMerge_urpmv13.safetensors --steps 30 --H 768 --W 512 --seed 0
done