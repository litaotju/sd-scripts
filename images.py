from PIL import Image
from pathlib import Path
from typing import *
from collections import OrderedDict
import os
import sys
import json

CONFIG_REQ_KEYS = ['Steps', 'Sampler', 'CFG scale', 'Seed', 'Size', 'Model hash']
CONFIG_OPT_KEYS = ['Model', 'Face restoration', 'AddNet Enabled', 'AddNet Module 1', 'AddNet Model 1', 'AddNet Weight A 1', 'AddNet Weight B 1']
class GenerationParameters(object):
    def __init__(self, prompt: str, negative_prompt: str, config: Dict[str, Any]):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.config = config

    def __str__(self):
        ret = self.prompt
        if self.negative_prompt != "":
            ret += os.linesep + "Negative prompt:" + self.negative_prompt 
        ret += os.linesep + ",".join(f"{k}: {v}" for k,v in self.config.items())
        return ret

def sanity_check_config(config: Dict[str, Any], check_opt=False) -> bool:
    '''
        return True if the config is valid, False otherwise.
    '''
    check_keys = CONFIG_REQ_KEYS
    check_keys += CONFIG_OPT_KEYS if check_opt else []
    missing_keys = [k for k in check_keys if k not in config]
    if len(missing_keys) > 0:
        print(f"Missing keys: {missing_keys}")
    return all([k in config for k in check_keys])

def extract_parameters_from_png(fpath: str) -> GenerationParameters:
    ''' Extract the "parameters" field from a png file, and return a GenerationParameters object.
        return None if the field is not found.
    '''
    assert isinstance(fpath, str)
    assert os.path.exists(fpath), f"Not exists file path: {fpath}"
    assert os.path.isfile(fpath), f"Not a file file path: {fpath}"
    assert os.path.splitext(fpath)[1] in ['.png'], f"Not a valid png file path: {fpath}"
    img = Image.open(fpath)

    parameter_str = img.info.get('parameters', "")
    NEGATIVE_PREFIX =  'Negative prompt:'
    if parameter_str:
        try:
            lines = parameter_str.split(os.linesep)
            prompt = lines[0]
            config_str = lines[-1]
            negative_prompt = ""
            if len(lines) > 2 and lines[1].startswith(NEGATIVE_PREFIX):
                negative_prompt = lines[1]
                negative_prompt = negative_prompt.split(NEGATIVE_PREFIX)[1].strip()
        except Exception:
            print(f"Error parsing {fpath}")
            print(parameter_str)
            raise Exception

        config = OrderedDict()
        for item in list(config_str.split(',')):
            k, v = item.split(':')
            config[k.strip()] = v.strip()
        p = GenerationParameters(prompt, negative_prompt, config)
        assert sanity_check_config(config), f"Sanity check failed for {fpath}, {config}, raw string: {parameter_str}"
        return p
    return None

def extract_png_info_to_json(out_json: str, fpaths: List[str]):
    file_to_params = []
    for fpath in fpaths:
        p = extract_parameters_from_png(fpath)
        d = OrderedDict()
        d['filename'] = fpath
        d['generation'] = p.__dict__ if p else {}
        file_to_params.append(d)
        d['cmd'] = parameter_to_cmd(p, "./") if p else ""

    with open(out_json, 'w') as f:
        json.dump(file_to_params, f, indent=4)

webui_sampler_to_cmd_sampler = {
    'Eular': 'euler',
    'Euler a': 'euler_a',
    #TODO: support other samplers
}

def parameter_to_cmd(p: GenerationParameters, out_dir) -> str:
    ''' Convert a GenerationParameters object to a command line string.
    '''
    sampler = webui_sampler_to_cmd_sampler.get(p.config['Sampler'], "ddim")
    ckpt = os.path.join(os.path.expanduser("~/stable-diffusion-webui/models/Stable-diffusion"), p.config['Model'] + ".safetensors")
    w, h = p.config['Size'].split('x')
    return f'''python ./gen_img_diffusers.py --prompt '{p.prompt}' --images_per_prompt 1 \
    --outdir {out_dir}  --fp16 --xformers --batch_size 1 \
    --ckpt {ckpt} \
    --steps {p.config['Steps']}  --W {w} --H {h} --seed {p.config['Seed']} \
    --sampler {sampler} --scale {p.config['CFG scale']} \
    '''
    #--network_module networks.lora --network_weights output/${model}/${model}${epoch}.safetensors  --network_mul ${strength} \

if __name__ == "__main__":
    out_json = sys.argv[1]
    fpaths = [f.strip() for f in sys.stdin]
    extract_png_info_to_json(out_json, fpaths)