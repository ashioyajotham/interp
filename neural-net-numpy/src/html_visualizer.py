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
                          'Activation Patterns', 'Network Architecture'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Add loss curve
        losses = [h['loss'] for h in self.training_history]
        fig.add_trace(go.Scatter(y=losses, name='Loss'), row=1, col=1)
        
        # Add weight distribution
        weights = self.training_history[-1]['weights']
        fig.add_trace(go.Histogram(x=weights.flatten(), name='Weights'), row=1, col=2)
        
        # Enhanced activation patterns
        activations = self.training_history[-1]['activations']
        fig.add_trace(
            go.Heatmap(
                z=activations,
                colorscale='Viridis',
                showscale=True,
                name='Activations'
            ), 
            row=2, col=1
        )
        
        # Network architecture visualization
        layer_sizes = [784] + [layer.weights.shape[0] for layer in self.layers]
        x_pos = np.linspace(0, 1, len(layer_sizes))
        y_pos = np.zeros(len(layer_sizes))
        
        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=y_pos,
                mode='markers+text',
                marker=dict(size=30, color='lightblue'),
                text=[f'Layer {i}<br>{s} units' for i, s in enumerate(layer_sizes)],
                textposition='top center',
                name='Layers'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            font=dict(size=14),
            title_font=dict(size=16),
            plot_bgcolor='white'
        )
        
        # Update axes
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=16, color='black')
        
        fig.write_html(self.dashboard_path)