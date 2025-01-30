import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.manifold import TSNE
import os
from datetime import datetime
plt.style.use('dark_background')

class NetworkVisualizer:
    def __init__(self, layers, save_dir='src/results'):
        self.layers = layers
        self.save_dir = save_dir
        self.training_losses = []
        self.activation_history = []
        plt.ion()  # Interactive mode
        self._create_dirs()
        
        # Create main figure with grid layout
        self.fig = plt.figure(figsize=(20, 10))
        self.gs = GridSpec(2, 3, figure=self.fig)
        
        # Initialize subplots
        self.weight_ax = self.fig.add_subplot(self.gs[0, 0])  # Weights
        self.act_ax = self.fig.add_subplot(self.gs[0, 1])     # Activations
        self.loss_ax = self.fig.add_subplot(self.gs[0, 2])    # Loss curve
        self.tsne_ax = self.fig.add_subplot(self.gs[1, :])    # t-SNE viz
        
        self.fig.tight_layout()
        plt.show()

    def _create_dirs(self):
        """Create directories for saving visualizations"""
        dirs = ['weights', 'activations', 'training']
        for d in dirs:
            os.makedirs(os.path.join(self.save_dir, d), exist_ok=True)

    def update(self, epoch, loss, input_data, hidden_states):
        """Update all visualizations"""
        self.training_losses.append(loss)
        self.activation_history.extend(hidden_states[-1])
        
        # Update weights
        self.weight_ax.clear()
        weights = self.layers[0].weights
        sns.heatmap(weights[:10, :10], ax=self.weight_ax, cmap='viridis')
        self.weight_ax.set_title('First Layer Weights Sample')
        
        # Update activations
        self.act_ax.clear()
        for i, state in enumerate(hidden_states):
            self.act_ax.plot(state[0, :20], label=f'Layer {i}')
        self.act_ax.set_title('Activation Patterns')
        self.act_ax.legend()
        
        # Update loss curve
        self.loss_ax.clear()
        self.loss_ax.plot(self.training_losses)
        self.loss_ax.set_title('Training Loss')
        self.loss_ax.set_yscale('log')
        
        # Update t-SNE visualization
        if epoch % 10 == 0 and len(self.activation_history) > 30:
            self.tsne_ax.clear()
            tsne = TSNE(n_components=2, perplexity=min(30, len(self.activation_history)-1))
            samples = np.array(self.activation_history[-100:])  # Last 100 samples
            projection = tsne.fit_transform(samples)
            self.tsne_ax.scatter(projection[:, 0], projection[:, 1], c='cyan', alpha=0.5)
            self.tsne_ax.set_title('Hidden Layer Representation')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def plot_weights(self, layer_idx=0):
        """Visualize weights as a heatmap"""
        weights = self.layers[layer_idx].weights
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights, cmap='viridis')
        plt.title(f'Layer {layer_idx} Weights')
        plt.show()

    def plot_activations(self, input_data):
        """Visualize activations through network"""
        activations = []
        output = input_data
        
        fig, axes = plt.subplots(1, len(self.layers), figsize=(15, 5))
        for i, layer in enumerate(self.layers):
            output = layer.forward(output)
            act_img = output.reshape(-1, int(np.sqrt(output.shape[1])))
            axes[i].imshow(act_img, cmap='viridis')
            axes[i].set_title(f'Layer {i} Activation')
        plt.show()

    def plot_digit_weights(self, layer_idx=0):
        """Visualize first layer weights as digit templates"""
        weights = self.layers[layer_idx].weights
        fig, axes = plt.subplots(8, 16, figsize=(20, 10))
        for i, ax in enumerate(axes.flat):
            if i < weights.shape[1]:
                ax.imshow(weights[:, i].reshape(28, 28), cmap='gray')
            ax.axis('off')
        plt.suptitle('First Layer Weight Patterns')
        plt.show()

    def save_weight_visualization(self, epoch):
        """Save weight matrices as heatmaps"""
        for i, layer in enumerate(self.layers):
            plt.figure(figsize=(10, 8))
            plt.imshow(layer.weights, cmap='viridis')
            plt.colorbar()
            plt.title(f'Layer {i} Weights - Epoch {epoch}')
            plt.savefig(f'{self.save_dir}/weights/layer_{i}_epoch_{epoch}.png')
            plt.close()

    def save_digit_templates(self, epoch):
        """Save first layer weights as digit templates"""
        weights = self.layers[0].weights
        plt.figure(figsize=(20, 10))
        for i in range(min(128, weights.shape[1])):
            plt.subplot(8, 16, i + 1)
            plt.imshow(weights[:, i].reshape(28, 28), cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Digit Templates - Epoch {epoch}')
        plt.savefig(f'{self.save_dir}/weights/digit_templates_epoch_{epoch}.png')
        plt.close()

    def visualize_weight_distributions(self, epoch):
        """Plot weight distributions for each layer"""
        plt.figure(figsize=(15, 5))
        for i, layer in enumerate(self.layers):
            plt.subplot(1, len(self.layers), i+1)
            sns.histplot(layer.weights.flatten(), kde=True)
            plt.title(f'Layer {i} Weights')
        plt.savefig(f'{self.save_dir}/interpretability/weight_dist_epoch_{epoch}.png')
        plt.close()

    def visualize_activations(self, input_data, epoch):
        """Visualize layer activations for given input"""
        activations = []
        output = input_data
        
        for i, layer in enumerate(self.layers):
            output = layer.forward(output)
            plt.figure(figsize=(10, 10))
            plt.imshow(output.reshape(int(np.sqrt(output.shape[1])), -1))
            plt.title(f'Layer {i} Activation Pattern')
            plt.savefig(f'{self.save_dir}/activations/layer_{i}_epoch_{epoch}.png')
            plt.close()

    def visualize_feature_importance(self, input_data, epsilon=1e-5):
        """Compute input feature importance using gradients"""
        original_output = input_data
        importance_map = np.zeros_like(input_data)
        
        for i in range(input_data.shape[1]):
            perturbed = input_data.copy()
            perturbed[0, i] += epsilon
            output = perturbed
            for layer in self.layers:
                output = layer.forward(output)
            importance_map[0, i] = np.abs(np.sum(output - original_output))
            
        plt.figure(figsize=(10, 10))
        plt.imshow(importance_map.reshape(28, 28), cmap='hot')
        plt.colorbar()
        plt.title('Feature Importance Map')
        plt.savefig(f'{self.save_dir}/interpretability/feature_importance.png')
        plt.close()

    def visualize_decision_boundary(self, X, y, epoch):
        """Project high-dimensional data to 2D and show decision boundaries"""
        # Get activations from last hidden layer
        hidden_output = X
        for layer in self.layers[:-1]:
            hidden_output = layer.forward(hidden_output)
            
        # Project to 2D using t-SNE
        tsne = TSNE(n_components=2)
        projection = tsne.fit_transform(hidden_output)
        
        plt.figure(figsize=(10, 10))
        plt.scatter(projection[:, 0], projection[:, 1], c=y)
        plt.title('Decision Boundary Visualization')
        plt.savefig(f'{self.save_dir}/interpretability/decision_boundary_epoch_{epoch}.png')
        plt.close()

    def visualize_weights(self, epoch):
        """Plot weight matrices for each layer"""
        for i, layer in enumerate(self.layers):
            plt.figure(figsize=(10, 8))
            plt.imshow(layer.weights, cmap='viridis')
            plt.colorbar()
            plt.title(f'Layer {i} Weights - Epoch {epoch}')
            plt.savefig(f'{self.save_dir}/weights/layer_{i}_epoch_{epoch}.png')
            plt.close()

    def visualize_activations(self, input_data, epoch):
        """Plot activation values through the network"""
        output = input_data
        for i, layer in enumerate(self.layers):
            output = layer.forward(output)
            plt.figure(figsize=(8, 6))
            # Plot as 1D activation pattern
            plt.plot(output.flatten())
            plt.title(f'Layer {i} Activations - Epoch {epoch}')
            plt.savefig(f'{self.save_dir}/activations/layer_{i}_epoch_{epoch}.png')
            plt.close()