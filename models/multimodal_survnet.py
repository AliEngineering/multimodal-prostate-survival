import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class MultimodalSurvNet(nn.Module):
    def __init__(self, clin_dim,
                 mri_out_dim=512,
                 clin_hidden=64,
                 fusion_hidden=128,
                 dropout=0.2,
                 freeze_mri=False):
        super().__init__()

        try:
            weights = R3D_18_Weights.KINETICS400_V1
            self.mri_encoder = r3d_18(weights=weights)
        except:
            self.mri_encoder = r3d_18(weights=None)

        self.mri_encoder.fc = nn.Identity()

        if freeze_mri:
            for p in self.mri_encoder.parameters():
                p.requires_grad = False

        self.clin_mlp = nn.Sequential(
            nn.Linear(clin_dim, clin_hidden),
            nn.BatchNorm1d(clin_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fusion = nn.Sequential(
            nn.Linear(mri_out_dim + clin_hidden, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1)
        )

    def forward(self, mri, clin):
        x_mri = self.mri_encoder(mri)
        x_clin = self.clin_mlp(clin)
        fused = torch.cat([x_mri, x_clin], dim=1)
        return self.fusion(fused).squeeze(-1)
