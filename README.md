# Text Classifier
## Introduction

This is a text classifier based on [BERT](https://arxiv.org/abs/1810.04805). This project is a PyTorch implementation of our work.

## Dependencies

```
pip install -r requirements.txt
```

## Quick Start for Fine-tuning

### Datasets

The raw data of our datasets are from the [KOBE](https://arxiv.org/abs/1903.12457). You can download the pre-processed datasets used in our paper on [Google Drive](https://drive.google.com/drive/folders/1xKaIHIm8TLBu6IlTVMFYX4zge2zM3lwi?usp=sharing).

### Fine-tuning

```shell
bash finetune_bert.sh
```

In the scripts, `--output_dir` denotes the directory to save the fine-tuning model. `--model_path` indicates the pre-trained checkpoint used for fine-tuning. You can refer to the fine-tuning codes for the description of other hyper-parameters.

### Inference

We also provide the inference scripts to directly acquire the generation results on the test sets.

```shell
bash infer_bert.sh
```

In the scripts, `--output_dir` denotes the directory of model checkpoint used for inference. The generated results are also saved in this directory.
