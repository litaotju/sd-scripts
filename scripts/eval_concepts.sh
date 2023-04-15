lora_weights=$1
prompts=$2
out=$3
mkdir $out/

ckpt=/home/litao/stable-diffusion-webui/models/Stable-diffusion/chilloutmix_NiPrunedFp32Fix.safetensors
negative_prompt="illustration, 3d, sepia, painting, cartoons, sketch, (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad hands, normal quality, ((monochrome)), ((grayscale:1.2)), futanari, full-package_futanari, penis_from_girl, newhalf, collapsed eyeshadow, multiple eyebrows, vaginas in breasts,holes on breasts, fleckles, stretched nipples, nipples on buttocks, analog, analogphoto, signatre, logo,2 faces"
strengths="0.6 0.8 0.9 1.0"

for strength in $strengths ; do
    prefix=$(basename ${lora_weights} .safetensors)-${strength}
    python ./gen_img_diffusers.py --from_file $prompts  --images_per_prompt 1 \
    --outdir ${out} --fp16 --xformers --batch_size 6 \
    --network_module networks.lora --network_weights ${lora_weights}  --network_mul ${strength} \
    --ckpt ${ckpt} --steps 30 --H 768 --W 512 --seed 42 --fname_prefix ${prefix} \
    --negative_prompt="${negative_prompt}"
done