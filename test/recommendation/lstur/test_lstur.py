import torch

from src.recommendation.lstur.LSTUR import LSTUR
from src.recommendation.lstur.PLMBasedNewsEncoder import PLMBasedNewsEncoder
from src.recommendation.lstur.UserEncoder import UserEncoder


def test_news_encoder() -> None:
    batch_size, max_length = 100, 30
    pretrained = "distilbert-base-uncased"
    conv_kernel_num, kernel_size = 300, 3
    attn_hidden_dim = 200

    input = torch.arange(batch_size * max_length).view(batch_size, max_length)

    news_encoder = PLMBasedNewsEncoder(pretrained, conv_kernel_num, kernel_size, attn_hidden_dim)

    assert tuple(news_encoder(input).size()) == (batch_size, conv_kernel_num)


def test_user_encoder() -> None:
    batch_size, max_length = 10, 30
    pretrained = "distilbert-base-uncased"
    conv_kernel_num, kernel_size = 300, 3
    user_num, history_size = 200, 3
    attn_hidden_dim = 200

    user_ids = torch.randint(0, user_num, (batch_size,))
    news_histories = torch.randint(0, 10, (batch_size, history_size, max_length))
    news_encoder = PLMBasedNewsEncoder(pretrained, conv_kernel_num, kernel_size, attn_hidden_dim)
    user_encoder = UserEncoder(conv_kernel_num, user_num)

    assert tuple(user_encoder(user_ids, news_histories, news_encoder).size()) == (batch_size, conv_kernel_num)


def test_lstur() -> None:
    batch_size, max_length = 10, 30
    pretrained = "distilbert-base-uncased"
    conv_kernel_num, kernel_size = 300, 3
    user_num, history_size = 200, 3
    attn_hidden_dim = 200
    news_candidate_num = 10
    vocab_num = 10

    news_encoder = PLMBasedNewsEncoder(pretrained, conv_kernel_num, kernel_size, attn_hidden_dim)
    user_encoder = UserEncoder(conv_kernel_num, user_num)

    # News Info
    news_candidates = torch.randint(0, vocab_num, (batch_size, news_candidate_num, max_length))
    # User Info
    user_ids = torch.randint(0, user_num, (batch_size,))
    news_histories = torch.randint(0, vocab_num, (batch_size, history_size, max_length))
    # Correct Labels(target)
    targets = torch.rand((batch_size, news_candidate_num))

    lstur = LSTUR(news_encoder, user_encoder)
    output = lstur(news_candidates, news_histories, user_ids, targets)  # [batch_size, news_candidate_num]
    assert tuple(output.logits.size()) == (batch_size, news_candidate_num)
