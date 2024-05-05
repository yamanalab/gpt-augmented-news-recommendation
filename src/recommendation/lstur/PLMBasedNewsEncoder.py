import torch
from transformers import AutoConfig, AutoModel

from recommendation.common_layers.AdditiveAttention import AdditiveAttention


class PLMBasedNewsEncoder(torch.nn.Module):
    def __init__(self, pretrained: str, conv_kernel_num: int, kernel_size: int, attn_hidden_dim: int = 200) -> None:
        super().__init__()
        self.plm = AutoModel.from_pretrained(pretrained)
        plm_hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size

        self.cnn = torch.nn.Conv1d(
            in_channels=plm_hidden_size, out_channels=conv_kernel_num, kernel_size=kernel_size, padding="same"
        )
        self.additive_attn = AdditiveAttention(conv_kernel_num, attn_hidden_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (batch_size, seq_len)
        e = self.plm(input).last_hidden_state  # (batch_size, seq_len, emb_dim)

        # inference CNN
        e = e.transpose(1, 2)  # (batch_size, emb_dim, seq_len)
        c = self.cnn(e)  # (batch_size, conv_kernel_num, seq_len)
        c = c.transpose(1, 2)  # (batch_size, seq_len, conv_kernel_num)

        # inference attn
        additive_attention_output = self.additive_attn(c)  # (batch_size, seq_len, conv_kernel_num)
        output = torch.sum(
            additive_attention_output, dim=1
        )  # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]

        return output
