import torch
import time
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

generator = torch.Generator(device="cuda").manual_seed(0)

def run():
    image = pipeline("An image of a squirrel in Picasso style", generator=generator).images[0]
    return image

# Warmup
warmup = 1
iteration = 5

for i in range(warmup):
    run()

start = time.perf_counter()
for i in range(iteration): run()
end = time.perf_counter()
print(f"Time to run 1 iteration: {(end - start)/iteration} s")
