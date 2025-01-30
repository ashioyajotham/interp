import torch
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class SAEAnalyzer:
    def __init__(self, model):
        self.model = model

    def compute_activation_statistics(self, dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std of neuron activations"""
        activations = []
        with torch.no_grad():
            for batch in dataloader:
                _, encoded = self.model(batch)
                activations.append(encoded)
        
        activations = torch.cat(activations, dim=0)
        return activations.mean(0), activations.std(0)

    def get_feature_directions(self) -> torch.Tensor:
        """Extract learned feature directions from encoder weights"""
        return self.model.encoder.weight.data

    def visualize_features(self, save_path: Optional[str] = None):
        """Visualize learned features"""
        weights = self.get_feature_directions()
        plt.figure(figsize=(12, 8))
        plt.imshow(weights.T.cpu().numpy(), cmap='RdBu', aspect='auto')
        plt.colorbar()
        plt.title("Feature Directions")
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def analyze_sparsity(self, input_data):
        with torch.no_grad():
            _, encoded = self.model(input_data)
            # Debug shape
            print(f"Encoded shape: {encoded.shape}")
            
            if len(encoded.shape) != 2:
                encoded = encoded.view(encoded.size(0), -1)
            
            total_activations = encoded.shape[0] * encoded.shape[1]
            active_neurons = torch.sum(encoded > 0).item()
            
            return {
                "sparsity": 1.0 - (active_neurons / total_activations),
                "active_neurons": active_neurons,
                "total_activations": total_activations
            }
