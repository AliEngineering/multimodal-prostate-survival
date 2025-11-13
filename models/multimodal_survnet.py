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

        # 3D ResNet-18 backbone (pretrained if possible)
        try:
            weights = R3D_18_Weights.KINETICS400_V1
            self.mri_encoder = r3d_18(weights=weights)
        except Exception:
            self.mri_encoder = r3d_18(weights=None)

        # replace final FC with identity to get 512-d embedding
        self.mri_encoder.fc = nn.Identity()

        if freeze_mri:
            for p in self.mri_encoder.parameters():
                p.requires_grad = False

        # Clinical branch (no BatchNorm -> LayerNorm so batch_size=1 is OK)
        self.clin_mlp = nn.Sequential(
            nn.Linear(clin_dim, clin_hidden),
            nn.LayerNorm(clin_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Fusion + survival head (again LayerNorm instead of BatchNorm)
        self.fusion = nn.Sequential(
            nn.Linear(mri_out_dim + clin_hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),  # risk score
        )

    def forward(self, mri, clin):
        """
        mri:  (B, 3, Z, Y, X)
        clin: (B, clin_dim)
        """
        x_mri = self.mri_encoder(mri)          # (B, mri_out_dim)
        x_clin = self.clin_mlp(clin)           # (B, clin_hidden)
        fused = torch.cat([x_mri, x_clin], dim=1)
        risk = self.fusion(fused).squeeze(-1)  # (B,)
        return risk
