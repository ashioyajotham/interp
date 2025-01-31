import plotly.express as px
import torch

def plot_activations(activations, threshold=0.5):
    """Plot activation patterns
    
    Args:
        activations: shape (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)
    """
    # Ensure 2D
    if len(activations.shape) == 3:
        activations = activations.squeeze(0)  # Remove batch dim
    
    # Transpose for better visualization
    act_matrix = activations.detach().numpy().T
    
    return px.imshow(
        act_matrix,
        title="Neuron Activations",
        labels={"x": "Token Position", "y": "Hidden Unit"},
        color_continuous_scale="viridis",
        aspect="auto"
    )

def plot_features(features):
    """Plot learned feature directions"""
    return px.imshow(
        features.detach().numpy(),
        title="Feature Directions",
        labels={"x": "Input Dimension", "y": "Feature"},
        color_continuous_scale="RdBu",
        aspect="auto"
    )

def plot_sparsity_dist(activations):
    """Plot activation distribution"""
    flat_acts = activations.flatten().detach().numpy()
    return px.histogram(
        flat_acts,
        title="Activation Distribution",
        labels={"value": "Activation", "count": "Frequency"},
        marginal="box"
    )