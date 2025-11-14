import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class MRIOnlySurvNet(nn.Module):
    def __init__(self,
                 mri_out_dim=512,
                 hidden=128,
                 dropout=0.2,
                 freeze_mri=False):
        super().__init__()

        try:
            weights = R3D_18_Weights.KINETICS400_V1
            self.mri_encoder = r3d_18(weights=weights)
        except Exception:
            self.mri_encoder = r3d_18(weights=None)

        self.mri_encoder.fc = nn.Identity()

        if freeze_mri:
            for p in self.mri_encoder.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(mri_out_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, mri):
        x = self.mri_encoder(mri)
        return self.head(x).squeeze(-1)
