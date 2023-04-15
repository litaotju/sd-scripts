person=$1
out=$2
lora_weights=$3
mkdir $out/

# ckpt=/home/litao/stable-diffusion-webui/models/Stable-diffusion/porn/uberRealisticPornMerge_urpmv13.safetensors 
# ckpt=/home/litao/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors
ckpt=/home/litao/stable-diffusion-webui/models/Stable-diffusion/chilloutmix_NiPrunedFp32Fix.safetensors
strength=0.9
prompts=${out}/prompts.txt

style="(8k, RAW photo, best quality, masterpiece:1.2), (realistic:1.3), (photorealistic:1.3), ultra-detailed, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3, best quality, beautiful lighting --n illustration, 3d, sepia, painting, cartoons, sketch, (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad hands, normal quality, ((monochrome)), ((grayscale:1.2)), futanari, full-package_futanari, penis_from_girl, newhalf, collapsed eyeshadow, multiple eyebrows, vaginas in breasts,holes on breasts, fleckles, stretched nipples, gigantic penis, nipples on buttocks, analog, analogphoto, signatre, logo,2 faces"

echo "a woman ${person} ${style}
${person} a woman with long hair wearing a pink top full-body shot ${style} --h 768 --w 512
a woman ${person}, headshot photo with sun glass looking back ${style} --h 768 --w 512
a photo of a woman ${person}, nude white female headshot with pink bra half-body shot ${style} --h 768 --w 512
woman ${person} standing by a table, holding an apple, with green hat ${style} --h 512 --w 768
a woman ${person} in a black lingerie posing for a picture in a mirror with her hands on her breasts ${style} --h 512 --768
" > ${prompts}

prefix=$(basename ${lora_weights} .safetensors)
python ./gen_img_diffusers.py --from_file $prompts  --images_per_prompt 1 \
--outdir ${out} --fp16 --xformers --batch_size 6 \
--network_module networks.lora --network_weights ${lora_weights}  --network_mul ${strength} \
--ckpt ${ckpt} --steps 30 --H 768 --W 512 --seed 42 --fname_prefix ${prefix}-${strength}