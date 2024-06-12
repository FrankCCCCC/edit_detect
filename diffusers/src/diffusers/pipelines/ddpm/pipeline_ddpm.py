# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union

import PIL
import PIL.Image
import numpy as np
import torch
from torchvision import transforms

from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput, ImagePipelineOutputExt


class DDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, inverse_scheduler=None):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.inverse_scheduler = inverse_scheduler

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        start_ratio_inference_steps: float = 0.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        init: torch.Tensor = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)
            
        if init is None:
            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                image = randn_tensor(image_shape, generator=generator)
                image = image.to(self.device)
            else:
                image = randn_tensor(image_shape, generator=generator, device=self.device)
        else:
            image = init.to(self.device)
            
        print(f"Image shape: {image.shape}, (min, max): ({torch.min(image)}, {torch.max(image)})")

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        pred_orig_images: list = []
        for t in self.progress_bar(self.scheduler.timesteps[-int(start_ratio_inference_steps * len(self.scheduler.timesteps)):]):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            output = self.scheduler.step(model_output, t, image, generator=generator)
            image = output.prev_sample
            pred_orig_images.append(output.pred_original_sample)

        pred_orig_images = torch.stack(pred_orig_images)
        latents = image.detach().cpu()
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutputExt(images=image, latents=latents, pred_orig_samples=pred_orig_images)
    
    @torch.no_grad()
    def invert(
        self,
        init: Union[torch.Tensor, np.ndarray, PIL.Image.Image],
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        start_ratio_inference_steps: float = 0.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe.invert().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)
            
        # if self.device.type == "mps":
        #     # randn does not work reproducibly on mps
        #     image = randn_tensor(image_shape, generator=generator)
        #     image = image.to(self.device)
        # else:
        #     image = randn_tensor(image_shape, generator=generator, device=self.device)
        
        if isinstance(init, np.ndarray):
            image = torch.from_numpy(init)
            if len(init.shape) < 4:
                image = torch.unsqueeze(image, 0)
        elif isinstance(init, torch.Tensor):
            image = init
            if len(init.shape) < 4:
                image = torch.unsqueeze(image, 0)
        elif isinstance(init, PIL.Image.Image):
            image = transforms.ToTensor()(init).unsqueeze(0)
        elif isinstance(init, list):
            if isinstance(init[0], np.ndarray):
                init = np.stack(init, axis=0)
                image = torch.from_numpy(init)
            elif isinstance(init[0], torch.Tensor):
                image = torch.stack(init, axis=0)
            elif isinstance(init[0], PIL.Image.Image):
                image = torch.stack([transforms.ToTensor()(e) for e in init], axis=0).permute(0, 3, 1, 2)
            else:
                raise TypeError(f"The elements of the arguement init should be numpy.ndarray, torch.Tensor, or PIL.Image.Image, not {type(init[0])}.")
        else:
            raise TypeError(f"Arguement init should be numpy.ndarray, torch.Tensor, or PIL.Image.Image, not {type(init)}.")
        
        print(f"Origianl image: {type(image)}, shape: {image.shape}, (min, max): ({torch.min(image)}, {torch.max(image)})")
        image = (image.to(self.device) - 0.5) * 2
        print(f"image: {type(image)}, shape: {image.shape}, (min, max): ({torch.min(image)}, {torch.max(image)})")

        # set step values
        self.inverse_scheduler.set_timesteps(num_inference_steps)

        pred_orig_images = []
        for t in self.progress_bar(self.inverse_scheduler.timesteps[-int(start_ratio_inference_steps * len(self.scheduler.timesteps)):]):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            step_output: ImagePipelineOutputExt = self.inverse_scheduler.step(model_output, t, image)
            image = step_output.prev_sample
            pred_orig_images.append(step_output.pred_orig_samples.detach().cpu())

        pred_orig_images = torch.stack(pred_orig_images)
        latents = image.detach().cpu()
        print(f"Latents shape: {latents.shape}, (min, max): ({torch.min(latents)}, {torch.max(latents)})")
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        # if not return_dict:
        #     return (image,)
        print(f"pred_orig_images: {pred_orig_images.shape}")

        return ImagePipelineOutputExt(images=image, latents=latents, pred_orig_samples=pred_orig_images)
