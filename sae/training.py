import torch
from torch.utils.data import DataLoader
from typing import Optional
from .autoencoder import SparseAutoencoder

def train_epoch(
    model: SparseAutoencoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        loss = model.get_reconstruction_loss(batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train(
    model: SparseAutoencoder,
    dataloader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> list[float]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    
    for epoch in range(epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return history