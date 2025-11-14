import torch
import torch.nn as nn


class ClinicalSurvNet(nn.Module):
    def __init__(self, clin_dim,
                 hidden=64,
                 fusion_hidden=128,
                 dropout=0.2):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(clin_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(fusion_hidden, 1),  # risk
        )

    def forward(self, clin):
        # clin: (B, clin_dim)
        return self.mlp(clin).squeeze(-1)  # (B,)
