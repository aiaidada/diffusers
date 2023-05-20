import argparse
import traceback
import shutil
import logging
import sys
import os
import torch
import numpy as np
import math
from pathlib import Path


from src.diffusers.pipelines import EstimDiffPipeline , DDIMPipeline


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
    parser.add_argument("--Skip_threshold", type= float, default= 0.002 ,
                         help="Threshold for increasing the number of steps")
    
    parser.add_argument("--Device", default= 'cpu' ,
                         help="Threshold for increasing the number of steps")

    parser.add_argument("--Mode", default= 'Normal' ,
                         help="Test or Noraml")   
    
    parser.add_argument("--Directory", default= None ,
                         help="Directory for saving generated images")  
    
    parser.add_argument("--Dataset" , default= "google/ddpm-church-256" , type= str, help= "Dataset for generated photo")

    #parser.add_argument() 
    args = parser.parse_args()


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
    model_id = args.Dataset
    ddim = EstimDiffPipeline.from_pretrained(model_id)
    ddim = ddim.to(device)
    im_num = args.Generation_num
    step_num = args.Step_num
    if not(args.Directory == None):
        path = Path(args.Directory)
    else:
        path = Path('Samples')
    path.mkdir(exist_ok=True)
    if args.skipping == 'Uniform':
        speed_up = 1000/step_num
        skip_num = math.ceil(2 * speed_up) - 2
        final = step_num - math.floor(2*1000/(skip_num+2))
        uniform = True
    else:
        uniform = False
        skip_num = 5
    if args.Mode == 'Normal':
        for i in range (im_num):
            image = torch.randn((1, 3 , 256 , 256) , device= device)
            im = ddim(num_inference_steps= 1000 , case_num= 47 , image= image , threshold= args.Skip_threshold , skip_num= skip_num , uniform= uniform , final= final)
            im['images'][0].save(os.path.join(str(path), "Sample_" + str(i) + ".png"))
    elif args.Mode == 'Test':
        original_path = Path(os.join(str(path) , 'DDIM(1000)'))
        original_path.mkdir(exist_ok= True)
        DDIM_path = Path(os.path(str(path) , '\DDIM('+str(step_num) + ')'))
        DDIM_path.mkdir(exist_ok= True)
        Proposed_path = Path(os.path.join(str(path) + '\Proposed('+str(step_num) + ')'))
        Proposed_path.mkdir(exist_ok= True)
        for i in range (im_num):
            image = torch.randn((1, 3 , 256 , 256) , device= device)
            im = ddim(num_inference_steps= 1000 , case_num= 47 , image= image , threshold= args.Skip_threshold , skip_num= skip_num , uniform= uniform , final= final)
            im['images'][0].save(os.path.join(str(Proposed_path), "Sample0_" + str(i) + ".png"))
            im = ddim(num_inference_steps= 1000 , case_num= 0 , image= image , threshold= args.Skip_threshold , skip_num= skip_num , uniform= uniform , final= final)
            im['images'][0].save(os.path.join(str(original_path), "Sample_" + str(i) + ".png"))
            im = ddim(num_inference_steps= step_num , case_num= 0 , image= image , threshold= args.Skip_threshold , skip_num= skip_num , uniform= uniform , final= final)
            im['images'][0].save(os.path.join(str(DDIM_path), "Sample_" + str(i) + ".png"))
    else:
        raise TypeError('Argument should only be Test or Noraml')


        

    return 0


if __name__ == "__main__":
    sys.exit(main())
