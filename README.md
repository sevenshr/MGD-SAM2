# MGD-SAM2
This repo is the official implementation for: 
[MGD-SAM2: Multi-view Guided Detail-enhanced Segment Anything Model 2 for High-Resolution Class-agnostic Segmentation](https://arxiv.org/abs/2503.23786).

## Abstract
Segment Anything Models (SAMs), as vision foundation models, have demonstrated remarkable performance across various image analysis tasks. Despite their strong generalization capabilities, SAMs encounter challenges in fine-grained detail segmentation for high-resolution class-independent segmentation (HRCS), due to the limitations in the direct processing of high-resolution inputs and low-resolution mask predictions, and the reliance on accurate manual prompts. To address these limitations, we propose MGD-SAM2 that integrates SAM2 with multi-view feature interaction between a global image and local patches to achieve precise segmentation. MGD-SAM2 incorporates the pre-trained SAM2 with four novel modules: the Multi-view Perception Adapter (MPAdapter), the Multi-view Complementary Enhancement Module (MCEM), the Hierarchical Multi-view Interaction Module (HMIM), and the Detail Refinement Module (DRM). Specifically, we first introduce MPAdapter to adapt the SAM2 encoder for enhanced extraction of local details and global semantics in HRCS images. Then, MCEM and HMIM are proposed to further exploit local texture and global context by aggregating multi-view features within and across multi-scales. Finally, DRM is designed to generate gradually restored high-resolution mask predictions, compensating for the loss of fine-grained details resulting from directly upsampling the low-resolution prediction maps.

## Overview
<img width="5013" height="1895" alt="fig1" src="https://github.com/user-attachments/assets/41044408-3139-4264-b3a4-f14f1b50e725" />

Here are some of our experiment results:
<img width="1141" height="1116" alt="fig3" src="https://github.com/user-attachments/assets/053948bd-5058-4b4d-bbf1-a7957c2f11d3" />

Here are some of our visual results:
<img width="5487" height="3503" alt="fig2" src="https://github.com/user-attachments/assets/4a68d7ba-79ce-47cf-a722-055ab88f2e93" />

## I. Requiremets
1. Clone this repository
```
git clone https://github.com/sevenshr/MGD-SAM2.git
cd MGD-SAM2
```

2.  Install packages

```
conda create -n mgdsam python=3.10
conda activate mgdsam
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

3. Download the corresponding dataset: four high-resolution datasets ([DIS5K](https://drive.google.com/file/d/1O1eIuXX1hlGsV7qx4eSkjH231q7G1by1/view?usp=sharing) for DIS, [HRSOD](https://github.com/yi94code/HRSOD), [DAVIS-S](https://github.com/yi94code/HRSOD) and [UHRSD](https://github.com/iCVTEAM/PGNet) for HRSOD) and two normal-resolution datasets ([DUTS](https://saliencydetection.net/duts/) and [HKU-IS](https://i.cs.hku.hk/~yzyu/research/deep_saliency.html) for SOD).
4. Download the pre-trained [SAM 2(Segment Anything)](https://github.com/facebookresearch/segment-anything-2) and put it in ./checkpoint.

## II. Training 
1. Update the data path in config file
2. Then, you can start training by simply running:

&nbsp;&nbsp;&nbsp;&nbsp;for **DIS** 
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --master_port=12459  --nnodes 1 --nproc_per_node 1 train.py  --config configs/dis-sam-vit-b1.yaml
```

&nbsp;&nbsp;&nbsp;&nbsp;for **HRSOD and NRSOD**
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --master_port=12459  --nnodes 1 --nproc_per_node 1 train.py  --config configs/hrsod-sam-vit-b1.yaml
```

## III. Testing 
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```
Download the pretrained model at [Google Drive](https://drive.google.com/drive/folders/1_y6uIFulnvdCG1EwcLfKuK56AO9HC85s?usp=sharing)

## Citation

If you find our work useful in your research, please consider citing:

```
@article{shen2025mgd,
  title={MGD-SAM2: Multi-view Guided Detail-enhanced Segment Anything Model 2 for High-Resolution Class-agnostic Segmentation},
  author={Shen, Haoran and Zhuang, Peixian and Kou, Jiahao and Zeng, Yuxin and Xu, Haoying and Li, Jiangyun},
  journal={arXiv preprint arXiv:2503.23786},
  year={2025}
}

```
## Thanks
Many thanks to the previous inspiring works:
```
@inproceedings{chen2023sam,
  title={Sam-adapter: Adapting segment anything in underperformed scenes},
  author={Chen, Tianrun and Zhu, Lanyun and Deng, Chaotao and Cao, Runlong and Wang, Yan and Zhang, Shangzhan and Li, Zejian and Sun, Lingyun and Zang, Ying and Mao, Papa},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3367--3375},
  year={2023}
}

@inproceedings{MVANet,
  title={Multi-view Aggregation Network for Dichotomous Image Segmentation},
  author={Yu, Qian and Zhao, Xiaoqi and Pang, Youwei and Zhang, Lihe and Lu, Huchuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3921--3930},
  year={2024}
}

```
