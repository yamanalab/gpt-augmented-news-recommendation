import torch
from recommendation.common_layers.AdditiveAttention import AdditiveAttention
from torch import nn


class UserEncoder(nn.Module):
    def __init__(self, conv_kernel_num: int, query_dim: int = 200) -> None:
        super().__init__()
        self.additive_attention = AdditiveAttention(conv_kernel_num, query_dim)

    def forward(self, news_histories: torch.Tensor, news_encoder: nn.Module) -> torch.Tensor:
        batch_size, hist_size, seq_len = news_histories.size()
        news_histories = news_histories.view(
            batch_size * hist_size, seq_len
        )  # (batch_size, hist_size, seq_len) -> (batch_size * hist_size, seq_len)

        news_histories_encoded = news_encoder(
            news_histories
        )  # (batch_size * hist_size, seq_len),(batch_size * hist_size,) -> (batch_size * hist_size, conv_kernel_num)
        conv_kernel_num = news_histories_encoded.size()[-1]

        news_histories_encoded = news_histories_encoded.view(
            batch_size, hist_size, conv_kernel_num
        )  # (batch_size * hist_size, conv_kernel_num) -> (batch_size, hist_size, conv_kernel_num)

        additive_attention_output = self.additive_attention(news_histories_encoded)
        # (batch_size, hist_size, conv_kernel_num), (batch_size, user_emb_dim) -> (batch_size, hist_size, conv_kernel_num)

        output = torch.sum(
            additive_attention_output, dim=1
        )  # (batch_size, seq_len, conv_kernel_num) -> (batch_size, conv_kernel_num)

        return output
