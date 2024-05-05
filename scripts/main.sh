#!/bin/bash

# source .envrc

# Train BERT Ours
rye run python src/experiment/train.py -m pretrained="bert-base-uncased" gradient_accumulation_steps=16 batch_size=8

# Train BERT baseline
rye run python src/experiment/train.py -m pretrained="bert-base-uncased" augmentation_method="rule_base" gradient_accumulation_steps=16 batch_size=8

# Shutdown instance after executiond
# shutdown -h now

