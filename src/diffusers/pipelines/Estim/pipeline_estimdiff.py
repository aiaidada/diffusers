# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import torch

from ...schedulers import DDIMScheduler
from ...utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class EstimDiffPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()

        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)
    
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 1000,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        case_num = 0,
        image= None, 
        threshold = 0.002,
        skip_num = 10,
        uniform = True,
        final = 3
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                if `True` or `False`, see documentation for `DDIMScheduler.step`. If `None`, nothing is passed
                downstream to the scheduler. So use `None` for schedulers which don't support this argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.sample_size, int):
            image_shape = (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size)
        else:
            image_shape = (batch_size, self.unet.in_channels, *self.unet.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        #image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        out = []
        l = []
        first_step = True
        Second_step = True
        
        print('Experimenting for case #' + str(case_num))
        if (case_num < 45):
            for t in self.progress_bar(self.scheduler.timesteps):
                ##################################################################
                ###########                  CASE 0                 ##############
                ##################################################################
                if case_num == 0:
                    model_output = self.unet(image, t).sample                                                    
                image = self.scheduler.step(
                    model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample    

        elif (case_num == 47):
            t= 999
            used = 0
            while (t >= 0):
                if (t < 3):
                    model_output = self.unet(image , t).sample
                    image = self.scheduler.step(
                        model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
                        ).prev_sample  
                    t = t -1 
                    #used += 1
                else:
                    model_output = self.unet(image , t).sample
                    if (len(out) > 1):
                        temp = torch.count_nonzero((((((model_output) > 0 )*((2*out[0] - out[1])<0))) + (((model_output) < 0 )*((2 * out[0] - out[1]) >0))))/(3*256*256)
                    else:
                        temp = 0 
                    if (len(out) < 2):
                        out.append(model_output)
                    out[0] = model_output
                    image = self.scheduler.step(
                        model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
                        ).prev_sample  
                    t = t -1 
                    model_output = self.unet(image , t).sample
                    if (len(out) < 2):
                        out.append(model_output)
                    out[1] =  model_output
                    image = self.scheduler.step(
                        model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
                        ).prev_sample 
                    #used += 2
                    t = t -1 
                    if (temp < threshold):
                        increase = True
                    else:
                        increase = False
                    for cou in range(int(skip_num)):
                        if (t < final):
                            continue
                        model_output = 2*out[1] - out[0]
                        out[0] = out[1]
                        out[1] = model_output
                        image = self.scheduler.step(
                        model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
                        ).prev_sample  
                        t  = t -1
                    if (increase and not(uniform)):
                        skip_num = skip_num + 5
                        
                    elif (skip_num > 8 and not(uniform)):
                        skip_num = skip_num - 2
                        
        

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,) 

        return ImagePipelineOutput(images=image) 

def count(model_output , out0 , out1):
    return(torch.count_nonzero(model_output))
    