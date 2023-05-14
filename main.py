import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

from src.diffusers.pipelines import EstimDiffPipeline


torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--skipping", type=str, required=True, help="Uniform or Guided Skipping"
    )
    
    parser.add_argument(
        "--Step_num",
        type= int,
        default= 1000,
        help="Number of steps",
    )
    parser.add_argument(
        "--Generation_num",
        type= int,
        default= 1,
        help="Number of images to generate",
    )
    parser.add_argument("--Skip_threshold", type= float, default= 0.002 , help="Threshold for increasing the number of steps")
    
    parser.add_argument("--Device", default= 'cpu' , help="Threshold for increasing the number of steps")

   
    args = parser.parse_args()



    
    
    
       








    # add device
   


    return args

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args = parse_args_and_config()
    device = torch.device("cuda") if (torch.cuda.is_available() and args.Device == 'gpu') else torch.device("cpu")
    model_id = "google/ddpm-church-256"
    ddim = EstimDiffPipeline.from_pretrained(model_id)
    ddim = ddim.to(device)
    im_num = args.Generation_num
    step_num = args.Step_num
    if args.skipping == 'Uniform':
        speed_up = 1000/step_num
        skip_num = (2 * speed_up) - 2
        uniform = True
    else:
        uniform = False
        skip_num = 5
    for i in range (im_num):
        image = torch.randn((1, 3 , 256 , 256) , device= device)
        im = ddim(num_inference_steps= 1000 , case_num= 47 , image= image , threshold= args.Skip_threshold , skip_num= skip_num , uniform= uniform)
        im['images'][0].save("Church" + str(i) + ".png") 

    return 0


if __name__ == "__main__":
    sys.exit(main())
