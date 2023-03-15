set -x

## Change this for your model's data and model name, and number of steps
input="./data/mkl.v1.0"
model_name="mkl.v1.0.2"
resume_model="mkl.v1.0.2"
resume_from="output/${resume_model}/${resume_model}-state"

steps=2000
epochs=100

resolution="512,512"
training_data_dir="data/mkl.v1.0/instances_512x512x100.cap/1_mkl"
base_model=/home/litao/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors

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
    --pretrained_model_name_or_path=${base_model} \
    --mixed_precision=fp16 \
    --train_batch_size=4 \
    --unet_lr=2e-4 \
    --text_encoder_lr=5e-5 \
    --optimizer_type=AdamW8bit \
    --network_module=networks.lora \
    --save_model_as=safetensors  \
    --clip_skip=2 --seed=42 \
    \
    --max_train_epochs=${epochs} \
    --max_train_steps=${steps} \
    --save_every_n_epochs=5 \
    --sample_every_n_epochs 3 \
    --save_last_n_epochs_state 1 \
    --save_state  \
    \
    --resolution=${resolution} --color_aug \
    \
    --sample_prompts="${input}/prompts.txt" \
    --train_data_dir=${training_data_dir} \
    --in_json=${input}/meta.json \
    ${REG_DATA_OPTION} \
    --output_name="${model_name}" \
    --output_dir=./output/${model_name} \
    --logging_dir=./output/${model_name} \
    --xformers \
    ${RESUME_OPTION}

cp ./output/${model_name}/${model_name}.safetensors  ~/stable-diffusion-webui/models/Lora

for strength in 0.0 0.2 0.4 0.6 0.8 1.0; do
    python ./gen_img_diffusers.py --from_file ${input}/prompts.txt --images_per_prompt 1 \
    --outdir output/${model_name}/generated/ --fp16 --xformers --batch_size 6 \
    --network_module networks.lora --network_weights output/${model_name}/${model_name}${epoch}.safetensors  --network_mul ${strength} \
    --ckpt ${base_model} --steps 30 --H 600 --W 400 --seed 0
done

set +x