set -x

## Change this for your model's data and model name, and number of steps
input="./data/dildo"
model_name="dildo.v0.2"
resume_model="dildo.v0.1"
resume_from="output/${resume_model}/${resume_model}-state"

steps=2000
epochs=100

resolution="512,768"
training_data_dir="data/dildo"
base_model=/home/litao/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors
base_model=/home/litao/stable-diffusion-webui/models/Stable-diffusion/uberRealisticPornMerge_urpmv13.safetensors 


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
    --lr_scheduler=cosine \
    --network_module=networks.lora \
    --save_model_as=safetensors  \
    --clip_skip=2 --seed=42000 \
    \
    --max_train_epochs=${epochs} \
    --max_train_steps=${steps} \
    --save_every_n_epochs=5 \
    --sample_every_n_epochs 3 \
    --save_last_n_epochs_state 1 \
    --save_state  \
    \
    --resolution=${resolution}  --random_crop --enable_bucket --color_aug \
    \
    --sample_prompts="${input}/prompts.txt" \
    --train_data_dir=${training_data_dir} \
    ${REG_DATA_OPTION} \
    --output_name="${model_name}" \
    --output_dir=./output/${model_name} \
    --logging_dir=./output/${model_name} \
    --xformers \
    ${RESUME_OPTION}

cp ./output/${model_name}/${model_name}.safetensors  ~/stable-diffusion-webui/models/Lora

### Run models on different epoch snapshots
ckpts="/home/litao/stable-diffusion-webui/models/Stable-diffusion/uberRealisticPornMerge_urpmv13.safetensors 
       /home/litao/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors"

strength=1.0
models=$(ls ./output/${model_name}/*safetensors)
for ckpt in ${ckpts}; do
for model in ${models}; do
    python ./gen_img_diffusers.py --from_file data/dildo/prompts.txt --images_per_prompt 1 \
    --outdir ./output/${model_name}/compare_epochs.upm  --fp16 --xformers --batch_size 6 \
    --network_module networks.lora --network_weights ${model} --network_mul ${strength} \
    --ckpt ${ckpt} --steps 30 --H 768 --W 512 --seed 0
done
done

set +x