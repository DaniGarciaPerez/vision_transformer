"""
"""

import torch
from torch import nn
import itertools
import plotly.express as px


class ScaledDotProductAttention(nn.Module):
    """ """

    def __init__(self, d_k):
        """ """
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """ """
        qk_matmul = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = torch.div(qk_matmul, (self.d_k**0.5))
        attention_weights = self.softmax(attention_scores)
        return torch.matmul(attention_weights, v)
