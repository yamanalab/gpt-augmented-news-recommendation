import torch
from recommendation.common_layers.PersonalizedAttention import PersonalizedAttention
from torch import nn


class UserEncoder(nn.Module):
    def __init__(self, conv_kernel_num: int, user_emb_dim: int, query_dim: int) -> None:
        super().__init__()
        self.personalized_attn = PersonalizedAttention(conv_kernel_num, user_emb_dim, query_dim)

    def forward(
        self, user_ids: torch.Tensor, news_histories: torch.Tensor, news_encoder: nn.Module, user_embedder: nn.Module
    ) -> torch.Tensor:
        assert len(user_ids) == len(news_histories)  # = batch_size

        batch_size, hist_size, seq_len = news_histories.size()
        news_histories = news_histories.view(
            batch_size * hist_size, seq_len
        )  # (batch_size, hist_size, seq_len) -> (batch_size * hist_size, seq_len)

        # Repeat user_ids: 
        # e.g.
        #   if history_size = 3, user_ids = {1,2,3,4} → {1,1,1,2,2,2,3,3,3,4,4,4}
        expanded_user_ids = user_ids.unsqueeze(0)  # (batch_size,) -> (1, batch_size)
        expanded_user_ids = expanded_user_ids.repeat(hist_size, 1)  # (1,batch_size) -> (hist_size, batch_size)
        expanded_user_ids = expanded_user_ids.transpose(1, 0)  # (hist_size, batch_size) -> (batch_size,hist_size)
        expanded_user_ids = expanded_user_ids.flatten()  # (hist_size, batch_size) -> (batch_size * hist_size,)
        news_histories_encoded = news_encoder(
            news_histories, expanded_user_ids, user_embedder
        )  # (batch_size * hist_size, seq_len),(batch_size * hist_size,) -> (batch_size * hist_size, conv_kernel_num)
        conv_kernel_num = news_histories_encoded.size()[-1]

        news_histories_encoded = news_histories_encoded.view(
            batch_size, hist_size, conv_kernel_num
        )  # (batch_size * hist_size, conv_kernel_num) -> (batch_size, hist_size, conv_kernel_num)

        user_embedding = user_embedder(user_ids)  # (batch_size,) -> (batch_size,user_emb_dim)

        personalized_attention_output = self.personalized_attn(news_histories_encoded, user_embedding)
        # (batch_size, hist_size, conv_kernel_num), (batch_size, user_emb_dim) -> (batch_size, hist_size, conv_kernel_num)

        output = torch.sum(
            personalized_attention_output, dim=1
        )  # (batch_size, seq_len, conv_kernel_num) -> (batch_size, conv_kernel_num)

        return output
