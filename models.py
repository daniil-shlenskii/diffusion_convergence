import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SimpleMLP(nn.Module):
    def __init__(self, d_in: int, T: int, hidden_dim: Optional[int]=128):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.x_proj = nn.Linear(d_in, self.hidden_dim)
        self.t_proj = nn.Embedding(T, self.hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * self.hidden_dim, d_in)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, t):
        '''
        :x input, e.g. images
        :t 1d th.gTensor of timesteps
        '''
        x = self.x_proj(x)
        t = self.t_proj(t.int())
        x = x + t
        x = F.gelu(x)
        return self.backbone(x)