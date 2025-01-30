import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, sparsity_param: float = 0.05):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_param = sparsity_param
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = torch.relu(self.encoder(x))
        # Apply sparsity via activation thresholding
        sparse_encoded = encoded * (encoded > self.sparsity_param).float()
        decoded = self.decoder(sparse_encoded)
        return decoded, sparse_encoded

    def get_reconstruction_loss(self, x: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
        decoded, encoded = self.forward(x)
        # MSE reconstruction loss
        recon_loss = nn.MSELoss()(decoded, x)
        # L1 sparsity penalty
        sparsity_loss = beta * torch.mean(torch.abs(encoded))
        return recon_loss + sparsity_loss