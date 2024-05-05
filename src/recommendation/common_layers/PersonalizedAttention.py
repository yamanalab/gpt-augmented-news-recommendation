import torch
from torch import nn


class PersonalizedAttention(nn.Module):
    def __init__(self, input_dim: int, user_emb_dim: int, query_dim: int) -> None:
        super().__init__()

        self.personalized_query_caluculater = nn.Sequential(
            nn.Linear(
                user_emb_dim, query_dim
            ),  # in: (batch_size, user_emb_dim), out: (batch_size, query_dim)
            nn.ReLU(),  # in: (batch_size, query_dim), out: (batch_size, query_dim)
        )

        self.linear = nn.Linear(query_dim, input_dim)  # linear to create query
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, c: torch.Tensor, user_embedding: torch.Tensor) -> torch.Tensor:

        personalized_query = self.personalized_query_caluculater(
            user_embedding
        )  # (batch_size, user_emb_dim) -> (batch_size, query_dim)

        """Calculate Attention Weight
        """
        weight = self.linear(
            personalized_query
        )  # (batch_size, query_dim) -> (batch_size, input_dim)
        weight = self.tanh(
            weight
        )  # # (batch_size, input_dim) -> (batch_size, input_dim)
        weight = weight.unsqueeze(
            -1
        )  # (batch_size, input_dim) -> (batch_size, input_dim, 1)
        attention_weight = torch.bmm(
            c, weight
        )  # [batch_size, seq_len, input_dim] x [batch_size, input_dim, 1] -> [batch_size, seq_len, 1]
        attention_weight = self.softmax(
            attention_weight
        )  # [batch_size, seq_len, 1] -> [batch_size, seq_len, 1]
        return c * attention_weight
