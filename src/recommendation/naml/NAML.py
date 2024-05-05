import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput


class NAML(nn.Module):
    def __init__(
        self,
        news_encoder: nn.Module,
        user_encoder: nn.Module,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
    ) -> None:
        super().__init__()
        self.news_encoder: nn.Module = news_encoder
        self.user_encoder: nn.Module = user_encoder
        self.loss_fn = loss_fn

    def forward(
        self, candidate_news: torch.Tensor, news_histories: torch.Tensor, user_id: torch.Tensor, target: torch.Tensor
    ) -> ModelOutput:
        """
        Encode Candidate News by News Encoder
        """
        batch_size, candidate_num, seq_len = candidate_news.size()
        candidate_news = candidate_news.view(
            batch_size * candidate_num, seq_len
        )  # [batch_size * candidate_num, seq_len]
        news_candidates_encoded = self.news_encoder(
            candidate_news
        )  # (batch_size * candidate_num, seq_len),(batch_size * candidate_num,) -> (batch_size * candidate_num, conv_kernel_num)
        conv_kernel_num = news_candidates_encoded.size()[-1]
        news_candidates_encoded = news_candidates_encoded.view(
            batch_size, candidate_num, conv_kernel_num
        )  # (batch_size * candidate_num, conv_kernel_num) -> (batch_size, hist_size, conv_kernel_num)

        """
        Encode User by User Encoder
        """
        user_encoded = self.user_encoder(
            news_histories, self.news_encoder
        )  # [batch_size,], [batch_size, histories, seq_len] -> [batch_size, conv_kernel_num]
        user_encoded = user_encoded.unsqueeze(-1)  # [batch_size, conv_kernel_num] -> [batch_size, conv_kernel_num, 1]
        output = torch.bmm(
            news_candidates_encoded, user_encoded
        )  # [batch_size, candidate_num, conv_kernel_num] x [batch_size, conv_kernel_num, 1] -> [batch_size, candidate_num, 1]
        output = output.squeeze(-1)

        # NOTE:
        # when "val" mode(self.training == False) → not calculate loss score
        # Multiple hot labels may exist on target.
        # e.g.
        # candidate_news = ["N24510","N39237","N9721"]
        # target = [0,2](=[1, 0, 1] in one-hot format)
        if not self.training:
            return ModelOutput(logits=output, loss=torch.Tensor([-1]), labels=target)

        loss = self.loss_fn(output, target)
        return ModelOutput(logits=output, loss=loss, labels=target)
