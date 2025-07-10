# DGA2
 Official Pytorch Implementation ``Fourier Guided Adaptive Adversarial Augmentation for Generalization in Visual Reinforcement Learning'' 

## Abstract
> Reinforcement learning (RL) has proven its potential in complex decision-making tasks. Yet, many RL systems rely on manually crafted state representations, requiring effort in feature engineering. 
Visual Reinforcement Learning (VRL) offers a way to address this challenge by enabling agents to learn directly from raw visual input. Nonetheless, VRL continues to face generalization issues, as models often overfit to specific domain features.
To tackle this issue, we propose Diffusion Guided Adaptive Augmentation (DGA2), an augmentation method that utilizes Stable Diffusion to enhance domain diversity.
We introduce an Adaptive Domain Shift strategy that dynamically adjusts the degree of domain shift according to the agent’s learning progress for effective augmentation with Stable Diffusion.
Additionally, we employ saliency as the mask to preserve the semantics of data.
Our experiments on the DMControl-GB, Adroit, and Procgen environments demonstrate that DGA2 improves generalization performance compared to existing data augmentation and generalization methods.



## Framework
[main_architecture.pdf](https://github.com/user-attachments/files/21153914/main_architecture.pdf)

## Experimental Results
### DMControl-GB
![dmcontrol](https://github.com/user-attachments/assets/c4c45265-968d-4c77-8eeb-60176b6cce91)

### Procgen
![procgen](https://github.com/user-attachments/assets/1f35fa6e-e37b-4006-a6de-8b2407cc0545)


### Adroit
![adroit](https://github.com/user-attachments/assets/dcecd333-7e9a-463c-9728-961f3e307d1d)


## Setup
### DMControl Guideline

#### Install MuJoCo
Download the MuJoCo version 2.0 binaries for Linux or OSX. 

#### Install DMControl

``` bash
conda env create -f setup/conda.yaml
conda activate fga3
sh setup/install_envs.sh
```
### Augmentation Guideline
To use Stable Diffusion for augmentation, you need to set up a separate environment and run the diffusion server independently before training the agent.
We provide a Flask-based API server to handle communication between the diffusion model and the RL training loop.

#### Diffusion Setup
``` bash
conda env create -f environment2.yml
conda activate diffusion
```


## Training
Note that the process of generating domain images using diffusion is separate from the main reinforcement learning framework. Therefore, it is necessary to execute the two Python scripts, make_image.py and train.py, sequentially.
Two processes communicate with Flask.
``` bash
# For Diffusion Generation
conda activate diffusion
CUDA_VISIBLE_DEVICES=0 python src/make_image.py --port 5000

# DMControl-GB
conda activate dmcgb
CUDA_VISIBLE_DEVICES=0 python src/train.py --domain_name walker --task_name walk --algorithm sac --seed 1111 --action_repeat 4 --port 5000 --mask_type exp --tag exp --train_steps 500k
