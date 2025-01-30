import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import wandb

class SAETrainer:
    def __init__(
        self, 
        model: nn.Module,
        lr: float = 0.001,
        l1_coef: float = 0.001,
        use_wandb: bool = False,
        wandb_config: dict = None
    ):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.l1_coef = l1_coef
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(
                project="sae-interpretability",
                config=wandb_config or {
                    "learning_rate": lr,
                    "l1_coefficient": l1_coef,
                    "model_type": model.__class__.__name__,
                    "input_dim": model.input_dim,
                    "hidden_dim": model.hidden_dim
                }
            )
            wandb.watch(model, log_freq=100)

    def loss_function(self, reconstructed, original, encoded):
        recon_loss = nn.MSELoss()(reconstructed, original)
        sparsity_loss = self.l1_coef * torch.norm(encoded, 1)
        return recon_loss + sparsity_loss, recon_loss, sparsity_loss

    def train_step(self, batch):
        self.optimizer.zero_grad()
        reconstructed, encoded = self.model(batch)
        loss, recon_loss, sparsity_loss = self.loss_function(reconstructed, batch, encoded)
        loss.backward()
        self.optimizer.step()
        return {
            'total_loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'sparsity_loss': sparsity_loss.item()
        }

    def train_epoch(self, dataloader, epoch: int):
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            losses = self.train_step(batch)
            epoch_losses.append(losses)
            
            if self.use_wandb:
                # Log detailed metrics
                wandb.log({
                    "epoch": epoch,
                    "batch": batch_idx,
                    **losses,
                    "activation_sparsity": self._compute_activation_sparsity(),
                    "weight_histogram": wandb.Histogram(self.model.encoder.weight.data.cpu()),
                    "activation_heatmap": wandb.Image(self._plot_activation_heatmap())
                })
        
        return epoch_losses

    def _compute_activation_sparsity(self):
        """Compute fraction of zero activations"""
        with torch.no_grad():
            sample_batch = next(iter(self.dataloader))
            _, encoded = self.model(sample_batch)
            return (encoded == 0).float().mean().item()

    def _plot_activation_heatmap(self):
        """Generate activation heatmap for visualization"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        with torch.no_grad():
            sample_batch = next(iter(self.dataloader))
            _, encoded = self.model(sample_batch)
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(encoded[0].cpu().numpy(), cmap='viridis')
            plt.title("Neuron Activations")
            plt.xlabel("Hidden Dimension")
            plt.ylabel("Sample")
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf)
            plt.close()
            buf.seek(0)
            return Image.open(buf)