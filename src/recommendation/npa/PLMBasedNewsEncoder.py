import torch
from recommendation.common_layers.PersonalizedAttention import PersonalizedAttention
from transformers import AutoConfig, AutoModel


class PLMBasedNewsEncoder(torch.nn.Module):
    def __init__(
        self,
        pretrained: str,
        conv_kernel_num: int,
        kernel_size: int,
        user_emb_dim: int,
        query_dim: int,
    ) -> None:
        super().__init__()
        self.plm = AutoModel.from_pretrained(pretrained)
        plm_hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size
        self.cnn = torch.nn.Conv1d(
            in_channels=plm_hidden_size,
            out_channels=conv_kernel_num,
            kernel_size=kernel_size,
            padding="same",
        )
        self.personalized_attn = PersonalizedAttention(
            conv_kernel_num, user_emb_dim, query_dim
        )

    def forward(
        self,
        input: torch.Tensor,
        user_ids: torch.Tensor,
        user_embedder: torch.nn.Module,
    ) -> torch.Tensor:
        # input: (batch_size, seq_len)
        e = self.plm(input).last_hidden_state  # (batch_size, seq_len, emb_dim)

        # inference CNN
        e = e.transpose(1, 2)  # (batch_size, emb_dim, seq_len)
        c = self.cnn(e)  # (batch_size, conv_kernel_num, seq_len)
        c = c.transpose(1, 2)  # (batch_size, seq_len, conv_kernel_num)

        user_embedding = user_embedder(
            user_ids
        )  # (batch_size,) -> # (batch_size, user_emb_dim)
        personalized_attention_output = self.personalized_attn(c, user_embedding)
        # (batch_size, seq_len, conv_kernel_num), (batch_size, user_emb_dim) -> (batch_size, seq_len, conv_kernel_num)
        output = torch.sum(
            personalized_attention_output, dim=1
        )  # (batch_size, seq_len, conv_kernel_num) -> (batch_size, conv_kernel_num)
        return output
