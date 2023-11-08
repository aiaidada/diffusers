<p align="center">
    <br>
    <img src="./docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue">
    </a>
    <a href="https://github.com/huggingface/diffusers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg">
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg">
    </a>
</p>

This is HuggingFace diffusers plus a new pipeline called Estim. This new very AWSOME pipeline, will generate photos very similar to thoes generated with 1000 steps. The trick is that without any training it decieds when to take the steps. It is cool. Trust me and TRY IT OUT.


## Prepration

You first have to install a few things (This goes if you want to use any of the diffusers pipelines)

    
```bash
pip install diffusers transformers accelerate scipy safetensors
```


## Modes

This pipeline has two mode for skipping the steps in diffusion models, the first one is the standard uniform skipping in which you can state the number of steps that you are willing to take. The second one on the other hand, decides the best number of steps, on the fly, based on different factors, and it can reduce the number of steps to as low as 40.


## Generating Images

This is same as any other pipeline, the only thing is that you can use Estim now. 


```python
from diffusers import EstimDiffPipeline


pipe = EstimDiffPipeline.from_pretrained("google/ddpm-cat-256").to("cuda")

im = pipe(num_inference_steps= 50  , threshold= args.Skip_threshold , uniform= True )
```

## Credits

This is the HuggingFace libarary and I just added the EstimDiff Pipline to it.

## Citation

```bibtex
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```
