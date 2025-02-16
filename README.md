# Depth-guided NeRF Training via Earth Mover's Distance

[![arXiv](https://img.shields.io/badge/Arxiv-2407.06189-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2403.13206)

[Anita Rau](https://anitarau.github.io), [Josiah Aklilu](https://www.linkedin.com/in/josiah-aklilu-876796167/), [F. Christopher Holsinger](https://med.stanford.edu/profiles/chris-holsinger),  [Serena Yeung-Levy](https://ai.stanford.edu/~syyeung/)

ECCV 2024

This work proposes a novel approach to addressing uncertainty in depth priors for NeRF supervision. 
Instead of using custom-trained depth or uncertainty priors, we use off-the-shelf pretrained diffusion 
models to predict depth and capture uncertainty during the denoising process. Because we know that depth 
priors are prone to errors, we propose to supervise the ray termination distance distribution with Earth 
Mover's Distance instead of enforcing the rendered depth to replicate the depth prior exactly through L2-loss. 
As a result, the geometric representation of a scene is dramatically more accurate.


## Getting Started
### Setting up the environment
TODO

### Data
Follow [this](https://github.com/barbararoessle/dense_depth_priors_nerf) repo to download the processed ScanNet data.

### Depth priors and uncertainty maps
Download our DiffDP depth priors and uncertainty maps [here](https://drive.google.com/drive/folders/1SwPUeiEMO2uE1LrETGytti0aSB-W8a2i?usp=share_link). Move them to the folder containing the ScanNet dataset you downloaded in the previous step.


## Training & Testing
```
. train_emdnerf.sh
```

### Testing
TODO

## 🙏 Acknowledgements
Our codebase is largely built on [SCADE](https://github.com/mikacuy/scade/tree/master) which is laregly built on [DDP](https://github.com/barbararoessle/dense_depth_priors_nerf)!

## ✏️ Citation
If you find our paper and code useful for your research, please consider citing us ✏️ and giving this repo a star ⭐️.
```BibTeX
@inproceedings{rau2024depth,
  title={Depth-guided nerf training via earth mover’s distance},
  author={Rau, Anita and Aklilu, Josiah and Holsinger, F Christopher and Yeung-Levy, Serena},
  booktitle={European Conference on Computer Vision},
  pages={1--17},
  year={2024},
  organization={Springer}
}
```