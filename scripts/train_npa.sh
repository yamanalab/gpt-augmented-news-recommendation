# rye run python src/experiment/train.py -m pretrained="distilbert-base-uncased","bert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method=GPT4 news_recommendation_model=NPA max_len=64

rye run python src/experiment/train.py -m pretrained="distilbert-base-uncased","bert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method=NONE news_recommendation_model=NAML max_len=64 epochs=3
rye run python src/experiment/train.py -m pretrained="distilbert-base-uncased" gradient_accumulation_steps=16 batch_size=8 augmentation_method=NONE news_recommendation_model=NPA max_len=64 epochs=3 random

