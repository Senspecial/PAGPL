import torch
from torch import nn


class GPromptDenoised(nn.Module):
    def __init__(self, input_dim):
        super(GPromptDenoised, self).__init__()
        self.input_dim = input_dim
        self.weight = nn.Parameter(torch.Tensor(1, input_dim))

        # Denoiser 模块：残差网络结构
        self.denoiser = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        # 门控模块（增强表达）
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        for layer in self.denoiser:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.gate:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, node_embeddings):

        fused_input = node_embeddings * self.weight

        noise = self.denoiser(fused_input)
        clean_prompt = fused_input - noise

        return clean_prompt