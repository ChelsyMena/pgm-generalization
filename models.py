import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TimeMLP(nn.Module):
    def __init__(self, dim=784, hidden_dim=512, time_embed_dim=128, dropout=0.1):
        super().__init__()
        
        # 1. High-fidelity Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim * 4),
        )

        # 2. Main Network
        self.input_proj = nn.Linear(dim, hidden_dim)
        
        # Time projection for Scale and Shift (AdaGN style)
        self.t_to_scale_shift = nn.Linear(time_embed_dim * 4, hidden_dim * 2)
        
        self.mid_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        self.output_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x, t):
        # Embed time
        t_emb = self.time_mlp(t)
        
        # Project data
        h = self.input_proj(x)
        
        # Scale-Shift Conditioning
        # This allows time to both multiply and add to the features
        t_params = self.t_to_scale_shift(t_emb)
        scale, shift = torch.chunk(t_params, 2, dim=-1)
        h = h * (1 + scale) + shift
        
        # Process and Output
        h = self.mid_layers(h)
        return self.output_proj(h)