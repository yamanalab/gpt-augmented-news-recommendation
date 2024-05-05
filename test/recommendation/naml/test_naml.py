import torch

from src.recommendation.naml.PLMBasedNewsEncoder import PLMBasedNewsEncoder
from src.recommendation.naml.UserEncoder import UserEncoder
from src.recommendation.naml.NAML import NAML


def test_news_encoder() -> None:
    batch_size, max_length = 100, 30
    pretrained = "distilbert-base-uncased"
    conv_kernel_num, kernel_size = 300, 3
    query_dim = 100

    input = torch.randint(0, 10, (batch_size, max_length))
    news_encoder = PLMBasedNewsEncoder(pretrained, conv_kernel_num, kernel_size, query_dim)

    assert tuple(news_encoder(input).size()) == (batch_size, conv_kernel_num)


def test_user_encder() -> None:
    batch_size, max_length = 100, 30
    pretrained = "distilbert-base-uncased"
    history_size = 2
    conv_kernel_num, kernel_size = 300, 3
    query_dim = 100

    news_histories = torch.randint(0, 10, (batch_size, history_size, max_length))

    news_encoder = PLMBasedNewsEncoder(pretrained, conv_kernel_num, kernel_size, query_dim)

    user_encoder = UserEncoder(conv_kernel_num, query_dim)

    output = user_encoder(news_histories, news_encoder)

    assert tuple(output.size()) == (batch_size, conv_kernel_num)


def test_npa() -> None:
    batch_size, max_length = 100, 30
    pretrained = "distilbert-base-uncased"
    history_size = 2
    news_candidate_num = 5
    conv_kernel_num, kernel_size = 300, 3
    user_num = 1000
    query_dim = 100

    news_encoder = PLMBasedNewsEncoder(pretrained, conv_kernel_num, kernel_size, query_dim)
    user_encoder = UserEncoder(conv_kernel_num, query_dim)

    news_candidates = torch.randint(0, 10, (batch_size, news_candidate_num, max_length))
    news_histories = torch.randint(0, 10, (batch_size, history_size, max_length))
    user_ids = torch.randint(0, user_num, (batch_size,))
    targets = torch.rand((batch_size, news_candidate_num))

    npa = NAML(news_encoder, user_encoder)

    output = npa(news_candidates, news_histories, user_ids, targets)  # [batch_size, news_candidate_num]

    assert tuple(output.logits.size()) == (batch_size, news_candidate_num)
