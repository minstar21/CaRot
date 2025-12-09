## Reproduction Project
Towards Calibrated Robust Fine-Tuning of Vision-Language Models (NeurIPS 2024)  
Changdae Oh*, Hyesu Lim*, Mijoo Kim, Dongyoon Han, Sangdoo Yun, Jaegul Choo, Alexander Hauptmann, Zhi-Qi Cheng^, Kyungwoo Song^

[arXiv](https://arxiv.org/abs/2311.01723)

---

## Table of Contents
- [Objective](#objective)
- [Experiments Conducted](#experiments-conducted)
- [Getting Started (Installation)](#getting-started-installation)
- [Add directory to PYTHONPATH](#add-directory-to-pythonpath)
- [Script to reproduce the main result](#script-to-reproduce-the-main-result)
- [Acknowledgement](#acknowledgement)

---

## Objective
Reproduce the results and understand the algorithm.

---

## Experiments Conducted

I carried out several experiments to reproduce and analyze the behavior of the original CaRot method:

- **Reproduced the results reported in Table 2 and Table 3** of the original paper.
- **Examined the effect of modifying the optimal hyperparameters (OC and SD)** as presented in Table 4, and evaluated how these changes impact final performance.
- **Evaluated model performance under different batch sizes** to analyze robustness to training configuration changes.

All experiment outputs, including logs, metrics, and per-setting results, are stored in the expt_logs directory.
<br/>

**Summary of Findings**
- The reproduced results were highly consistent with the original paper, showing only about a 0.5% difference in accuracy and a 0.016 difference in ECE compared to the reported performance.
- While OC and SD showed generally similar variation patterns, the SD coefficient produced the most noticeable differences, closely matching the trends reported in Table 4 of the original paper.
- Due to the inherent behavior of CLIP-based models, the proposed method showed performance degradation at smaller batch sizes, consistent with trends commonly observed in contrastive vision–language training.

---

## Getting Started (Installation)

```bash
git clone https://github.com/minstar21/CaRot.git
cd CaRot
conda create -n CaRot python=3.10
conda activate CaRot
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
mkdir checkpoints # The checkpoints directory is used to store model weights generated during fine-tuning

```

### Add directory to PYTHONPATH:

```bash
cd CaRot
export PYTHONPATH="$PYTHONPATH:$PWD"
```


### Script to reproduce the main result
* Refer to the DATA.md for the ImageNet directory strucutre.

```bash
# Create required dataset folders
mkdir -p datasets/data
mkdir -p datasets/csv

ln -s /your_dataset_location/imagenet ./datasets/data/ILSVRC2012
ln -s /your_dataset_location/imagenet_R ./datasets/data/imagenet-r
ln -s /your_dataset_location/imagenet_A ./datasets/data/imagenet-a
ln -s /your_dataset_location/imagenet_S ./datasets/data/sketch
ln -s /your_dataset_location/objectnet-1.0 ./datasets/data/objectnet-1.0

# Create ImageNet CSV for fine-tuning
python datacreation_scripts/imagenet_csv_creator.py

OC=0.2 # The optimal hyperparameter value for the orthogonality constraint
SD=1.5 # The optimal hyperparameter value for the self-distillation

python src/main.py \
--train-dataset=ImageNet --epochs=10 --lr 1e-5 --wd 0.1 --batch-size 256 \ --grad-accum-steps 2 \
--model=ViT-B/16 --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch,ObjectNet \
--template=openai_imagenet_template  --save=./checkpoints/ \
--data-location=./datasets/data/ --ft_data="./datasets/csv/imagenet.csv" \
--csv-img-key filepath --csv-caption-key title --exp_name ImageNet/carot \
--cross_fnorm 0.05 --l_orth_wv $OC --distil_coef $SD \
--wb_project "None" --method carot
```

<br/>

---

<br/>

### Acknowledge
This repository is built on top of the [CaRot](https://github.com/MLAI-Yonsei/CaRot) project.
I independently reproduced the original work and sought to replicate its reported results.
