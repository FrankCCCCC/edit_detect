# !pip install diffusers
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddpm: DDPMPipeline = DDPMPipeline.from_pretrained(model_id).to('cuda')  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

image_list: list = []
# run pipeline in inference (sample random noise and denoise)
for i in range(4):
    image = ddpm(batch_size=32).images
    image_list += image

# save image
for i, image in enumerate(image_list):
    image.save(f"fake_images/cifar10_ddpm/horse{i}.jpg")
