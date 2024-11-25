# Robust Surgical Phase Recognition From Annotation Efficient Supervision
This is the official code used for "Robust Surgical Phase Recognition From Annotation Efficient Supervision".

[![arXiv](https://img.shields.io/badge/arxiv-2207.00449-red)](https://arxiv.org/abs/2406.18481)

This repo was forked from https://github.com/ZheLi2020/TimestampActionSeg, which corresponds to the paper
[" Temporal Action Segmentation from Timestamp Supervision](https://arxiv.org/abs/2103.06669).

We thank the authors for releasing their codes. 


## Feature Extraction

Dataset preparation and the feature extraction code is detailed [here](https://github.com/or7r/SelfSupSurg-Based-Feature-Extraction).  


Next, reformat the extracted features using `scripts/reformat_features.py` to extract features.

```
python scripts/reformat_features.py --feature_extract_path "path to feature extraction folder" --dataset cholec80
python scripts/reformat_features.py --feature_extract_path "path to feature extraction folder" --dataset MultiBypass140
```

## Installation

You need to have a Anaconda3 installed for the setup. We developed the code on Ubuntu 20.04 system, CUDA 11.3 using A100 GPUs.



```
conda create -n "segmentation-stable" python=3.8.13 ipython
conda activate segmentation-stable
conda install mamba -c conda-forge
mamba install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
mamba install -c conda-forge gxx_linux-64
mamba install -c conda-forge wandb=0.12.21 numpy xlsxwriter tqdm matplotlib gcc=10.4.0

pip install cmake loguru tensorboard optuna argparse pyyaml focal_loss_torch scipy scikit-image scikit-learn
```

Next, clone and build [gtn](https://github.com/facebookresearch/gtn?tab=readme-ov-file#installation) from source and clone [gtn_applications](https://github.com/facebookresearch/gtn_applications.git).



### Training:

```
# Model training
config_file=#Path to config file
python "main.py" --config_file "$config_file"

# Psuedo-label Generation
certanity_temp=0.25
python main.py --action predict --save_certain --quiet --config_file "$config_file" --save_certain_temp $certanity_temp

additional_config_file=#Path to config file of additional training
python "main.py" --config_file "$additional_config_file"

```


### Citation:

If you use the code, please cite

    @misc{rubin2024robustsurgicalphaserecognition,
      title={Robust Surgical Phase Recognition From Annotation Efficient Supervision}, 
      author={Or Rubin and Shlomi Laufer},
      year={2024},
      eprint={2406.18481},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.18481}, 
}
    
