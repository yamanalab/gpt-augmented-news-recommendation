<h1 align="center"> News Recommendation with <br /> Category Description via Large Language Model </h1>


## Overview

<div align="center">
    <img src="./.github/images/method-overview.png" alt="Overview of our proposed method.">
</div>

This repository is the official implementation for the paper: **News Recommendation with Category Description via Large Language Model**. 

In this study, we proposed a novel approach that utilizes Large Language Models (LLMs) to automatically generate descriptive texts for news categories, which are then applied to enhance news recommendation performance. Comprehensive experiments demonstrate that our proposed method achieves a 5.8% improvement in performance compared to baselines.

## Directories
```bash
$ tree -L 2
.
├── LICENSE
├── README.md
├── dataset/ # MIND dataset and its download script
│   ├── download_mind.py
│   ├── generated/
│   └── mind/
├── pyproject.toml
├── requirements-dev.lock
├── requirements.lock
├── scripts/ # Script (Bash) for the experiment
│   ├── train_naml.sh
│   ├── train_npa.sh
│   └── train_nrms.sh
├── src/
│   ├── config/ # Configuration
│   ├── const/
│   ├── evaluation/ # Evaluation Metrics: nDCG, AUC, MRR
│   │   ├── RecEvaluator.py
│   ├── experiment/ # Experiment Command
│   │   ├── generation/
│   │   └── train.py
│   ├── mind/ # Loading the dataset
│   │   ├── CategoryAugmentedMINDDataset.py
│   │   ├── MINDDataset.py
│   │   └── dataframe.py
│   ├── recommendation/ # Recommendation Models by PyTorch & Transformers
│   │   ├── __init__.py
│   │   ├── common_layers/
│   │   ├── naml/
│   │   ├── npa/
│   │   └── nrms/
│   └── utils/
└── test/ # Unit test
    ├── evaluation
    ├── mind
    └── recommendation
```

## Requirements

- [Rye](https://rye-up.com/) 

It also works with Python v3.11.3 + pip.

## Setting

At first, you can install dependencies by running: 

```bash
$ rye sync
```

Next, please set PYTHONPATH to environment variable:

```bash
$ export PYTHONPATH=$(pwd)/src:$(pwd)
```

## Download MIND dataset

We use **[MIND (Microsoft News Dataset)](https://msnews.github.io/)** dataset for training and validating the news recommendation model. You can download them by executing [dataset/download_mind.py](https://github.com/YadaYuki/news-recommendation-llm/blob/main/dataset/download_mind.py).


```bash
$ rye run python ./dataset/download_mind.py 
```

By executing [dataset/download_mind.py](https://github.com/YadaYuki/news-recommendation-llm/blob/main/dataset/download_mind.py), the MIND dataset will be downloaded from an external site and then extracted.

If you successfully executed, `dataset` folder will be structured as follows:

```
./dataset/
├── download_mind.py
└── mind
    ├── large
    │   ├── test
    │   ├── train
    │   └── val
    ├── small
    │   ├── train
    │   └── val
    └── zip
        ├── MINDlarge_dev.zip
        ├── MINDlarge_test.zip
        ├── MINDlarge_train.zip
        ├── MINDsmall_dev.zip
        └── MINDsmall_train.zip
```

## Generate Category Description by GPT-4

In this step, you will need an **OpenAI API_KEY**. Please follow [this document](https://platform.openai.com/docs/quickstart) to obtain an API_KEY. 

If you are unable to issue an API_KEY, the category descriptions generated in this step are already provided in the repository ([**category_description_gpt4.json**](https://github.com/yamanalab/gpt-augmented-news-recommendation/blob/main/dataset/generated/category_description_gpt4.json)), so please use those.

At first, please set OpenAI API_KEY to environment variable:

```bash
$ export OPENAI_API_KEY={YOUR_OPENAI_API_KEY}
```

Then, generate category description by running:

```bash
$ rye run python ./src/experiment/generation/gpt_based_text_generation.py
```

After running this scripts, you can confirm that the file `category_description_gpt4.json` has been generated under the `dataset/generated` directory.