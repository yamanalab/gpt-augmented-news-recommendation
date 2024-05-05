import torch

from src.recommendation.common_layers.AdditiveAttention import AdditiveAttention
from src.recommendation.common_layers.PersonalizedAttention import PersonalizedAttention


def test_additive_attention() -> None:
    batch_size, seq_len, emb_dim, hidden_dim = 20, 10, 30, 5
    attn = AdditiveAttention(emb_dim, hidden_dim)
    input = torch.rand(batch_size, seq_len, emb_dim)
    assert tuple(attn(input).shape) == (batch_size, seq_len, emb_dim)


def test_personalized_attention() -> None:
    batch_size, seq_len, conv_kernel_num = 20, 10, 30
    query_dim = 5
    user_emb_dim = 100

    input = torch.rand(batch_size, seq_len, conv_kernel_num)
    user_embeddings = torch.rand(batch_size, user_emb_dim)

    attn = PersonalizedAttention(conv_kernel_num, user_emb_dim, query_dim)
    output = attn(input, user_embeddings)

    assert tuple(output.size()) == (batch_size, seq_len, conv_kernel_num)
