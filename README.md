# Grounding-REVERIE-Challenge
This repository is the official REVERIE Referring Expression Grounding model of [REVERIE-Challenge](https://yuankaiqi.github.io/REVERIE_Challenge/).
This code derives from [UNITER](https://github.com/ChenRocks/UNITER), fine-tuned on [REVERIE data](https://github.com/YuankaiQi/REVERIE/tree/master/tasks/REVERIE/data_v2). If you have any question, please do not hesitate to contact me, chongyang.zhao@adelaide.edu.au.


## Requirements
You need [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) to run the code. For more details, please refer to [UNITER](https://github.com/ChenRocks/UNITER) for further README information.  We tested on Ubuntu 20.04 and Nvidia 2080ti.
  - [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
  - [Docker group membership](https://docs.docker.com/engine/install/linux-postinstall/)

## Data Preparation
Note that BBoxes_v2 files have two different forms of organization, so you need download [Grounding_BBoxes_v2](https://drive.google.com/drive/folders/1nEaScjwGaIP3r_LtGnheUGqbFBGy1VSt?usp=sharing), which is different from the one used by [HOP-REVERIE-Challenge](https://github.com/YanyuanQiao/HOP-REVERIE-Challenge).

Download the [REVERIE data](https://github.com/YuankaiQi/REVERIE/tree/master/tasks/REVERIE/data_v2), [ Grounding_BBoxes_v2](https://drive.google.com/drive/folders/1nEaScjwGaIP3r_LtGnheUGqbFBGy1VSt?usp=sharing) and object features ([reverie_obj_feats_v2.pkl](https://drive.google.com/file/d/1zwV3QDPUVt7YmBNqTaCdS6v01U4b6p7M/view?usp=sharing) or [reverie_bbox_feat_v2_caffe.zip]( https://pan.baidu.com/s/1hxNypQZLz21RQpMD6yQNag?pwd=nubg), code: nubg), both of them are extracted using [BUTD Faster R-CNN](https://github.com/peteanderson80/bottom-up-attention) trained on [Visual Genome](http://visualgenome.org/), and organise data like below:
```
|- Grounding-REVERIE-Challenge
    |- Downloads
        |- REVERIE
        |   |- REVERIE_val_seen.json
        |   |- REVERIE_val_unseen.json
        |   |- REVERIE_test.json
        |                     
        |- Grounding_BBoxes_v2
        |   |- 1LXtFkjw3qL_0b22fa63d0f54a529c525afbf2e8bb25.json
        |   |- 1LXtFkjw3qL_0b302846f0994ec9851862b1d317d7f2.json
        |   ...           
        |   |- zsNo4HB9uLZ_faad06c7cb2b4a6f9220e7f6f87c800b.json
        |
        |- reverie_bbox_feat_v2_caffe (option 1)
        |   |- 1LXtFkjw3qL_0b22fa63d0f54a529c525afbf2e8bb25_00.h5
        |   |- 1LXtFkjw3qL_0b22fa63d0f54a529c525afbf2e8bb25_01.h5
        |   ...           
        |   |- zsNo4HB9uLZ_faad06c7cb2b4a6f9220e7f6f87c800b_35.h5
        |
        |- reverie_obj_feats_v2.pkl (option 2)
```
```bash
# Image object features preparation
# Make sure the parameters in script are correct
# 
# Support .h5 and .pkl files
# .h5:
# feats_path: "Downloads/reverie_bbox_feat_v2_caffe"
# feats_format: "h5"
# .pkl:
# feats_path: "Downloads/reverie_obj_feats_v2.pkl"
# feats_format: "pkl"

bash img_data_preparation.sh
```

## Pre-trained Model
- Grounding pre-trained model weights: `weights/ckpt/model_epoch_best.pt`
  - Download the `model_epoch_best.pt` from [here](https://drive.google.com/drive/folders/1nEaScjwGaIP3r_LtGnheUGqbFBGy1VSt?usp=sharing).

## Usage
1. ```bash
    # Docker image should be automatically pulled

    cd Grounding-REVERIE-Challenge
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
    # boxes_dir: "Downloads/Grounding_BBoxes_v2"
    #
    # We support multi-GPU inference
    # NUM_GPUS=4

    bash eval.sh
    ```
