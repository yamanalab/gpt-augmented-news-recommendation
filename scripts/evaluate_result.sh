#!/bin/bash


# Train Template-based
## NRMS
rye run python src/experiment/train.py -m pretrained="bert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method="gpt4" news_recommendation_model="nrms"
rye run python src/experiment/train.py -m pretrained="distilbert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method="gpt4" news_recommendation_model="nrms"

## NPA
rye run python src/experiment/train.py -m pretrained="bert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method="gpt4" news_recommendation_model="npa"
rye run python src/experiment/train.py -m pretrained="distilbert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method="gpt4" news_recommendation_model="npa"

## LSTUR
rye run python src/experiment/train.py -m pretrained="bert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method="gpt4" news_recommendation_model="lstur"
rye run python src/experiment/train.py -m pretrained="distilbert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method="gpt4" news_recommendation_model="lstur"


# Train Ours
## NRMS
rye run python src/experiment/train.py -m pretrained="bert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method="template_based" news_recommendation_model="nrms"
rye run python src/experiment/train.py -m pretrained="distilbert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method="template_based" news_recommendation_model="nrms"

## NPA
rye run python src/experiment/train.py -m pretrained="bert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method="template_based" news_recommendation_model="npa"
rye run python src/experiment/train.py -m pretrained="distilbert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method="template_based" news_recommendation_model="npa"

## LSTUR
rye run python src/experiment/train.py -m pretrained="bert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method="template_based" news_recommendation_model="lstur"
rye run python src/experiment/train.py -m pretrained="distilbert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method="template_based" news_recommendation_model="lstur"
