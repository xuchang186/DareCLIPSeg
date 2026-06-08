# DareCLIPSeg

DareCLIPSeg provides the code used for data efficiency experiments on BTMRI and Kvasir.

## Environment

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The experiments were run with a CUDA-enabled PyTorch environment. The main dependency versions used in this repository are listed in `requirements.txt`, including:

```text
torch==2.9.0
torchvision==0.24.0
transformers==4.30.0
numpy<2.0
monai
timm
opencv-python
openpyxl
```

## Pretrained Weights

The data efficiency experiments use the UniMedCLIP ViT-B/16 pretrained visual-language backbone.

Place the pretrained weights and text encoder files under:

```text
checkpoints/
|-- unimed_clip_vit_b16.pt
|-- BiomedNLP-BiomedBERT-base-uncased-abstract/
|   |-- config.json
|   |-- pytorch_model.bin
|   |-- tokenizer_config.json
|   |-- vocab.txt
|   |-- special_tokens_map.json
```

The visual-language checkpoint version used for training is:

```text
UniMedCLIP ViT-B/16
checkpoint file: unimed_clip_vit_b16.pt
model config: ViT-B-16-quickgelu
```

The text encoder used for tokenization and text feature extraction is:

```text
BiomedNLP-BiomedBERT-base-uncased-abstract
```

Automatic downloading of pretrained weights is disabled. The code expects the files to exist locally before training starts.

## Datasets

Please obtain datasets yourself and place them under `data/`.

The required directory structure is:

```text
data/
|-- BTMRI/
|   |-- Train_Folder/
|   |   |-- img/
|   |   |-- label/
|   |-- Val_Folder/
|   |   |-- img/
|   |   |-- label/
|   |-- Test_Folder/
|   |   |-- img/
|   |   |-- label/
|   |-- Prompts_Folder/
|       |-- Train_text.xlsx
|       |-- Val_text.xlsx
|       |-- Train_text_10.xlsx
|       |-- Val_text_10.xlsx
|       |-- Train_text_25.xlsx
|       |-- Val_text_25.xlsx
|       |-- Train_text_50.xlsx
|       |-- Val_text_50.xlsx
|       |-- Test_text_original.xlsx
|-- Kvasir/
|   |-- Train_Folder/
|   |   |-- img/
|   |   |-- label/
|   |-- Val_Folder/
|   |   |-- img/
|   |   |-- label/
|   |-- Test_Folder/
|   |   |-- img/
|   |   |-- label/
|   |-- Prompts_Folder/
|       |-- Train_text.xlsx
|       |-- Val_text.xlsx
|       |-- Train_text_10.xlsx
|       |-- Val_text_10.xlsx
|       |-- Train_text_25.xlsx
|       |-- Val_text_25.xlsx
|       |-- Train_text_50.xlsx
|       |-- Val_text_50.xlsx
|       |-- Test_text_original.xlsx
```

The Excel files in `Prompts_Folder` must contain these columns:

```text
Image
Ground Truth
Description
```

## Training And Evaluation

Run one data efficiency experiment with:

```bash
bash scripts/efficiency.sh <output_dir> <dataset> <data_percentage> <seed>
```

Example:

```bash
bash scripts/efficiency.sh output BTMRI 25 42
```

Supported datasets:

```text
BTMRI
Kvasir
```

Supported data percentages:

```text
10
25
50
100
```

The script performs training, testing, and metric evaluation in sequence.

The output structure is:

```text
output/
|-- BTMRI_25/
|   |-- trained_models/
|   |-- seg_results/
|   |-- unc_results/
|-- Kvasir_25/
|   |-- trained_models/
|   |-- seg_results/
|   |-- unc_results/
```

Set the GPU with:

```bash
GPU_ID=0 bash scripts/efficiency.sh output Kvasir 100 42
```
