# model.py  **********************************

import torch
import torch.nn as nn
import torch.nn.functional as F


# FocalLoss *************************

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, preds, targets):
        ce_loss    = F.cross_entropy(preds, targets,
                                     weight=self.weight,
                                     reduction='none')
        pt         = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# FederatedMultimodalModel *************************

class FederatedMultimodalModel(nn.Module):
    def __init__(self, ts_input_size=5, tab_input_size=2,
                 hidden_size=128, num_classes=5, dropout=0.3):
        super().__init__()

        # Time-series encoder (LSTM)
        self.ts_encoder = nn.LSTM(
            ts_input_size, hidden_size,
            num_layers=2, batch_first=True, dropout=dropout
        )

        # Tabular encoder (MLP)
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_size)
        )

        # Attention fusion
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_size)

        # Local personalized heads (never shared)
        self.local_cls_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        self.local_rul_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_ts, x_tab):
        _, (h_n, _) = self.ts_encoder(x_ts)
        ts_emb      = h_n[-1]

        tab_emb     = self.tab_encoder(x_tab)

        tokens      = torch.stack([ts_emb, tab_emb], dim=1)
        fused, _    = self.attention(tokens, tokens, tokens)
        embedding   = self.norm(fused.mean(dim=1))

        cls_out     = self.local_cls_head(embedding)
        rul_out     = self.local_rul_head(embedding).squeeze(-1)

        return cls_out, rul_out, embedding

    def backbone_state(self):
        exclude = {'local_cls_head', 'local_rul_head'}
        return {k: v.clone() for k, v in self.state_dict().items()
                if not any(e in k for e in exclude)}

    def local_head_state(self):
        include = {'local_cls_head', 'local_rul_head'}
        return {k: v.clone() for k, v in self.state_dict().items()
                if any(e in k for e in include)}