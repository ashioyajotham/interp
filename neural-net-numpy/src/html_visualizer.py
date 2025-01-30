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
        # Create subplot figure
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
        
        # Save/update single dashboard
        fig.write_html(self.dashboard_path)