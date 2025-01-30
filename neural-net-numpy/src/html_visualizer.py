import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from datetime import datetime
import plotly.express as px

class HTMLVisualizer:
    def __init__(self, layers, save_dir='results'):
        self.layers = layers
        self.save_dir = save_dir
        self.training_history = []
        # Single dashboard path
        self.dashboard_path = os.path.join(save_dir, 'dashboard.html')
        
    def update(self, epoch, loss, weights, activations):
        """Update training history and generate HTML report"""
        self.training_history.append({
            'epoch': epoch,
            'loss': loss,
            'weights': weights,
            'activations': activations
        })
        
        # Always update single dashboard file
        if epoch % 10 == 0:
            self._generate_html()
    
    def _generate_html(self):
        """Generate single HTML dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Weight Distribution', 
                          'Activation Patterns', 'Network Architecture')
        )
        
        # Add loss curve
        losses = [h['loss'] for h in self.training_history]
        fig.add_trace(go.Scatter(y=losses, name='Loss'), row=1, col=1)
        
        # Add weight distribution
        weights = self.training_history[-1]['weights']
        fig.add_trace(go.Histogram(x=weights.flatten(), name='Weights'), 
                     row=1, col=2)
        
        # Add activation patterns
        activations = self.training_history[-1]['activations']
        fig.add_trace(go.Heatmap(z=activations), row=2, col=1)
        
        # Add network architecture
        layer_sizes = [784] + [layer.weights.shape[0] for layer in self.layers]
        layer_names = ['Input'] + [f'Dense {i+1}' for i in range(len(self.layers))]
        
        # Create nodes for each layer
        x_positions = np.linspace(0, 1, len(layer_sizes))
        y_positions = np.zeros_like(x_positions)
        
        # Add nodes (layers)
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=y_positions,
                mode='markers+text',
                marker=dict(size=20),
                text=layer_names,
                textposition='top center',
                hovertext=[f'Size: {size}' for size in layer_sizes],
                name='Layers'
            ),
            row=2, col=1
        )
        
        # Add edges (connections)
        for i in range(len(layer_sizes)-1):
            fig.add_trace(
                go.Scatter(
                    x=[x_positions[i], x_positions[i+1]],
                    y=[y_positions[i], y_positions[i+1]],
                    mode='lines',
                    line=dict(width=1),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(showlegend=True, height=800)
        fig.write_html(self.dashboard_path)