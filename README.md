# Grounding-REVERIE-Challenge
This repository is the official REVERIE Referring Expression Grounding model of [REVERIE-Challenge](https://yuankaiqi.github.io/REVERIE_Challenge/).
This code derives from [UNITER](https://github.com/ChenRocks/UNITER).


## Requirements
- [Nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation)
- [Nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart)
- [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- [Docker group membership](https://docs.docker.com/install/linux/linux-postinstall/)

For more details, please follow [UNITER](https://github.com/ChenRocks/UNITER) instructions. We tested on Ubuntu 20.04 and Nvidia 2080ti.

## Data Preparation
Download the [REVERIE data](https://github.com/YuankaiQi/REVERIE/tree/master/tasks/REVERIE/data_v2) and object features ([everie_obj_feats_v2.pkl](https://www.baidu.com) or [reverie_bbox_feat_v2_caffe.zip](https://www.baidu.com), both of them are extracted using [Bottom-up attention](https://github.com/peteanderson80/bottom-up-attention) Faster R-CNN), and organise data like below:
```
|- Grounding-REVERIE-Challenge
    |- Downloads
        |- REVERIE
        |   |- REVERIE_val_seen.json
        |   |- REVERIE_val_unseen.json
        |   |- REVERIE_test.json
        |                     
        |- BBoxes_v2
        |   |- 1LXtFkjw3qL_0b22fa63d0f54a529c525afbf2e8bb25.json
        |   |- 1LXtFkjw3qL_0b302846f0994ec9851862b1d317d7f2.json
        |   ...           
        |   |- zsNo4HB9uLZ_faad06c7cb2b4a6f9220e7f6f87c800b.json
        |
        |- reverie_bbox_feat_v2_caffe
        |   |- 1LXtFkjw3qL_0b22fa63d0f54a529c525afbf2e8bb25_00.h5
        |   |- 1LXtFkjw3qL_0b22fa63d0f54a529c525afbf2e8bb25_01.h5
        |   ...           
        |   |- zsNo4HB9uLZ_faad06c7cb2b4a6f9220e7f6f87c800b_35.h5
        |
        |- reverie_obj_feats_v2.pkl
```
```bash
# Image object features preparation
# Make sure the parameters in script are correct
# 
# Support .h5 and .pkl files
# .h5:
# feats_path: "Downloads/reverie_bbox_feat_v2_caffe/"
# feats_format: "h5"
# .pkl:
# feats_path: "Downloads/reverie_obj_feats_v2.pkl"
# feats_format: "pkl"

bash img_data_preparation.sh
```

## Weights
- Grounding model weights: `weights/ckpt/model_epoch_best.pt`
  - Download the `model_epoch_best.pt` from [here](https://drive.google.com/drive/folders/1nEaScjwGaIP3r_LtGnheUGqbFBGy1VSt?usp=sharing).

## Usage
1. ```bash
    # Docker image should be automatically pulled

    bash launch_container.sh
    ```
2. ```bash
    # Object id prediction 
    # Make sure the parameters in the script are correct
    #
    # Navigation Results:
    # input_nav_dir: "input_nav_dir"
    #
    # Grounding Results (Navigation Results + predObjId):
    # output_dir: "submit_file"
    #
    # reverie_dir: "Downloads/REVERIE"
    # boxes_dir: "Downloads/BBoxes_v2"
    #
    # We support multi-GPU inference
    # NUM_GPUS=4

    bash eval.sh
    ```
