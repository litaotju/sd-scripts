#! /usr/bin/bash

lora_weights=$(realpath $1)
workdir=$(dirname "${lora_weights}")
default_strengths="1.0"
strengths=${2:-"$default_strengths"}

default_ckpt=/home/litao/stable-diffusion-webui/models/Stable-diffusion/chilloutmix_NiPrunedFp32Fix.safetensors
ckpt=${3:-"${default_ckpt}"}

default_prompts=${workdir}/data/prompts.txt
prompts=${4:-"${default_prompts}"}
IMGAE_PER_PROMPT=4

model_abbr=$(echo $(basename ${ckpt}) | cut -c -8)
model_hash=$(stat ${ckpt} | grep "Modify" | sha1sum | cut -c -8)
tag="${model_abbr}-${model_hash}-st-${strengths}"

roles=("" "20 years young asign girl" "20 years young white woman" "super woman" "nier 2b" "emma watson")
negative_prompt="illustration, 3d, sepia, painting, cartoons, sketch, (worst quality:2), (low quality:2), \
(normal quality:2), lowres, bad anatomy, bad hands, normal quality, ((monochrome)), ((grayscale:1.2)), futanari, \
full-package_futanari, newhalf, collapsed eyeshadow, multiple eyebrows, vaginas in breasts,holes on breasts, \
fleckles, stretched nipples, nipples on buttocks, analog, analogphoto, signatre, logo,2 faces"

function print_prompts()
{
    trigger=$1
    for strength in $strengths; do
        for res in "--h 768 --w 512" "--h 512 --w 768"; do
            for r in ${!roles[@]}; do
                role="${roles[$r]}"
                echo "$trigger $role ${res} --am ${strength}" | tr -s ' '
            done
        done
    done
}

# =====================================================================
# auto set
if [ "Z${prompts}" = "Z" ] || [ ! -f ${prompts} ]; then
    echo "Did not find the prompts file ${prompts}, use the default one"
    prompts=${default_prompts}
fi

if [ ! -e ${lora_weights} ]; then
    echo "Can not find the base model ${lora_weights}"
    exit 1
fi

ts=$(date '+%Y-%m-%d-%H-%M-%S')
out=${workdir}/gen-${tag}-${ts}

if [ -e ${out} ]; then
    echo "The output dir ${out} already exists, please remove it first"
    exit 1
fi
mkdir $out/

lines=$(awk -F "-" '{print $1}' < ${prompts} | sort -u | awk NF)
while IFS= read -r line; do
    print_prompts "$line" >> ${out}/z_prompts.txt
done <<< "$lines"

prefix=$(basename ${lora_weights} .safetensors)
python ./gen_img_diffusers.py --from_file ${out}/z_prompts.txt  --images_per_prompt ${IMGAE_PER_PROMPT} \
--outdir ${out} --fp16 --xformers --batch_size 4 \
--network_module networks.lora --network_weights ${lora_weights} \
--ckpt ${ckpt} --steps 30 --H 768 --W 512 --seed 42 --fname_prefix ${prefix} \
--max_embeddings_multiples=3 \
--negative_prompt="${negative_prompt}" \
--restore_face