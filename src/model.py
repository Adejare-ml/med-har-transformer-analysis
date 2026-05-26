import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based architecture for activity classification.
    """
    def __init__(self, input_dim: int, model_dim: int = 64, num_heads: int = 4, 
                 num_layers: int = 2, num_classes: int = 7):
        super(TimeSeriesTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)
