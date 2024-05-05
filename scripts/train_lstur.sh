# LSTUR

rye run python src/experiment/train.py -m pretrained="distilbert-base-uncased" gradient_accumulation_steps=8 batch_size=8 augmentation_method=GPT4 news_recommendation_model=LSTUR max_len=64 learning_rate=0.001,0.0005,0.0002


rye run python src/experiment/train.py -m pretrained="distilbert-base-uncased" gradient_accumulation_steps=8 batch_size=8 augmentation_method=GPT4,NONE,TEMPLATE_BASED news_recommendation_model=LSTUR max_len=64 learning_rate=0.0001 

rye run python src/experiment/train.py -m pretrained="bert-base-uncased" gradient_accumulation_steps=8 batch_size=8 augmentation_method=GPT4,NONE,TEMPLATE_BASED news_recommendation_model=LSTUR max_len=64 learning_rate=0.0001 