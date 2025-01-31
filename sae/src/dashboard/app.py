import streamlit as st
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.model_loader import ModelLoader
from src.models.autoencoder import SparseAutoencoder
from src.evaluation.analyzer import SAEAnalyzer

class SAEDashboard:
    def __init__(self):
        self.initialize_page()
        self.setup_state()
        
    def initialize_page(self):
        st.set_page_config(layout="wide")
        st.title("Sparse Autoencoder Interpretability")
        
    def setup_state(self):
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'activations' not in st.session_state:
            st.session_state.activations = None
            
    def sidebar_config(self):
        with st.sidebar:
            st.header("Model Configuration")
            configs = {
                "model": st.selectbox("Model", ["bert", "gpt2"]),
                "layer": st.slider("Layer", -12, -1, -1),
                "hidden_dims": st.slider("Hidden Dimensions", 32, 256, 64),
                "sparsity": st.slider("Sparsity", 0.0, 1.0, 0.1),
                "threshold": st.slider("Activation Threshold", 0.0, 1.0, 0.5)
            }
            return configs
            
    def neuron_inspector(self, activations, tokens):
        st.subheader("Neuron Analysis")
        
        # Neuron selector with importance score
        importance = torch.norm(activations, dim=0)
        top_neurons = torch.argsort(importance, descending=True)
        
        neuron_id = st.selectbox(
            "Select Neuron",
            top_neurons.tolist(),
            format_func=lambda x: f"Neuron {x} (Importance: {importance[x]:.3f})"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            # Interactive activation timeline
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=activations[:, neuron_id].numpy(),
                x=tokens,
                mode='lines+markers',
                hovertemplate="Token: %{x}<br>Activation: %{y:.3f}"
            ))
            fig.update_layout(
                title=f"Neuron {neuron_id} Activation Pattern",
                xaxis_title="Token Position",
                yaxis_title="Activation"
            )
            st.plotly_chart(fig)
            
        with col2:
            # Token attribution with highlighting
            token_scores = activations[:, neuron_id].numpy()
            fig = px.bar(
                x=tokens,
                y=token_scores,
                title="Token Attribution",
                labels={"x": "Token", "y": "Attribution Score"}
            )
            st.plotly_chart(fig)
            
    def feature_analysis(self, sae, activations):
        st.subheader("Feature Analysis")
        
        # Feature selection
        features = sae.encoder.weight.data
        selected_features = st.multiselect(
            "Select Features",
            range(features.shape[0]),
            default=[0, 1, 2]
        )
        
        if selected_features:
            # Feature correlation matrix
            corr = torch.corrcoef(activations[:, selected_features].T)
            fig = px.imshow(
                corr.numpy(),
                title="Feature Correlations",
                labels={"color": "Correlation"}
            )
            st.plotly_chart(fig)
            
            # Feature importance distribution
            importance = torch.norm(features[selected_features], dim=1)
            fig = px.bar(
                x=selected_features,
                y=importance.numpy(),
                title="Feature Importance Distribution"
            )
            st.plotly_chart(fig)
            
    def metrics_dashboard(self, analyzer, activations):
        st.subheader("Analysis Metrics")
        
        # Sparsity analysis
        stats = analyzer.analyze_sparsity(activations)
        
        cols = st.columns(3)
        cols[0].metric("Sparsity", f"{stats['sparsity']:.3f}")
        cols[1].metric("Active Neurons", stats['active_neurons'])
        cols[2].metric("Dead Neurons", stats['total_neurons'] - stats['active_neurons'])
        
        # Distribution plots
        fig = px.histogram(
            activations.flatten().numpy(),
            title="Activation Distribution",
            marginal="box"
        )
        st.plotly_chart(fig)
    
    def run(self):
        configs = self.sidebar_config()
        
        text = st.text_area("Input Text", "The quick brown fox jumps over the lazy dog")
        
        if text:
            model = ModelLoader(configs["model"])
            activations = model.get_activations(text, configs["layer"])
            tokens = model.tokenizer.tokenize(text)
            
            sae = SparseAutoencoder(
                input_dim=activations.shape[-1],
                hidden_dim=configs["hidden_dims"],
                sparsity_param=configs["sparsity"]
            )
            
            analyzer = SAEAnalyzer(sae)
            
            tabs = st.tabs(["Neuron Inspector", "Feature Analysis", "Metrics"])
            
            with tabs[0]:
                self.neuron_inspector(activations.squeeze(0), tokens)
            with tabs[1]:
                self.feature_analysis(sae, activations.squeeze(0))
            with tabs[2]:
                self.metrics_dashboard(analyzer, activations.squeeze(0))

if __name__ == "__main__":
    dashboard = SAEDashboard()
    dashboard.run()
