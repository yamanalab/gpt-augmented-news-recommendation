import torch
from torch import nn


class UserEncoder(nn.Module):
    def __init__(
        self,
        conv_kernel_num: int,  # = conv_kernel_num
        user_num: int,
        dropout: float = 0.2,  # same as: https://aclanthology.org/P19-1033.pdf
        user_representation_mask_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(user_num, conv_kernel_num)
        self.ltur_mask_layer = nn.Dropout(p=user_representation_mask_prob)
        self.gru = nn.GRU(conv_kernel_num, conv_kernel_num, batch_first=True)

    def forward(self, user_ids: torch.Tensor, news_histories: torch.Tensor, news_encoder: nn.Module) -> torch.Tensor:
        assert len(user_ids) == len(news_histories)  # = batch_size

        # LTUR: Long term user representation
        user_embeddings = self.user_embeddings(user_ids)  # [batch_size] -> [batch_size, hidden_size]
        user_embeddings = self.ltur_mask_layer(
            user_embeddings
        )  # [batch_size, hidden_size] -> [batch_size, hidden_size]
        user_embeddings = user_embeddings.unsqueeze(0)  # [batch_size, hidden_size] -> [1, batch_size, hidden_size]

        # STUR: Short term user representation
        batch_size, hist_size, seq_len = news_histories.size()
        news_histories = news_histories.view(batch_size * hist_size, seq_len)
        news_encoded = news_encoder(news_histories)
        emb_dim = news_encoded.size()[-1]
        news_histories_encoded = news_encoded.view(batch_size, hist_size, emb_dim)  #
        _, h_final = self.gru(news_histories_encoded, user_embeddings)  #
        h_final = h_final.squeeze(0)

        return h_final
