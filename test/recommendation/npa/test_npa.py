import torch

from src.recommendation.npa.NPA import NPA
from src.recommendation.npa.PLMBasedNewsEncoder import PLMBasedNewsEncoder
from src.recommendation.npa.UserEncoder import UserEncoder


def test_news_encoder() -> None:
    batch_size, max_length = 100, 30
    pretrained = "distilbert-base-uncased"
    conv_kernel_num, kernel_size = 300, 3
    user_num, user_emb_dim = 1000, 50
    query_dim = 100

    input = torch.randint(0, 10, (batch_size, max_length))
    user_ids = torch.randint(0, user_num, (batch_size,))
    news_encoder = PLMBasedNewsEncoder(pretrained, conv_kernel_num, kernel_size, user_emb_dim, query_dim)
    user_embedder = torch.nn.Embedding(user_num, user_emb_dim)

    assert tuple(news_encoder(input, user_ids, user_embedder).size()) == (batch_size, conv_kernel_num)


def test_user_encder() -> None:
    batch_size, max_length = 100, 30
    pretrained = "distilbert-base-uncased"
    history_size = 2
    conv_kernel_num, kernel_size = 300, 3
    user_num, user_emb_dim = 1000, 50
    query_dim = 100

    news_histories = torch.randint(0, 10, (batch_size, history_size, max_length))
    user_ids = torch.randint(0, user_num, (batch_size,))

    news_encoder = PLMBasedNewsEncoder(pretrained, conv_kernel_num, kernel_size, user_emb_dim, query_dim)
    user_embedder = torch.nn.Embedding(user_num, user_emb_dim)

    user_encoder = UserEncoder(conv_kernel_num, user_emb_dim, query_dim)

    output = user_encoder(user_ids, news_histories, news_encoder, user_embedder)

    assert tuple(output.size()) == (batch_size, conv_kernel_num)


def test_npa() -> None:
    batch_size, max_length = 100, 30
    pretrained = "distilbert-base-uncased"
    history_size = 2
    news_candidate_num = 5
    conv_kernel_num, kernel_size = 300, 3
    user_num, user_emb_dim = 1000, 50
    query_dim = 100

    news_encoder = PLMBasedNewsEncoder(pretrained, conv_kernel_num, kernel_size, user_emb_dim, query_dim)
    user_encoder = UserEncoder(conv_kernel_num, user_emb_dim, query_dim)

    news_candidates = torch.randint(0, 10, (batch_size, news_candidate_num, max_length))
    news_histories = torch.randint(0, 10, (batch_size, history_size, max_length))
    user_ids = torch.randint(0, user_num, (batch_size,))
    targets = torch.rand((batch_size, news_candidate_num))

    npa = NPA(news_encoder, user_encoder, user_num, user_emb_dim)

    output = npa(news_candidates, user_ids, news_histories, targets)  # [batch_size, news_candidate_num]

    assert tuple(output.logits.size()) == (batch_size, news_candidate_num)
