import streamlit as st
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.evaluation.analyzer import SAEAnalyzer

class AnalysisComponent:
    def __init__(self, sae, activations, tokens=None):
        self.sae = sae
        self.activations = activations
        self.tokens = tokens
        self.analyzer = SAEAnalyzer(sae)
        
    def render_activation_analysis(self):
        st.subheader("Activation Analysis")
        
        # Sparsity metrics
        stats = self.analyzer.analyze_sparsity(self.activations)
        cols = st.columns(3)
        
        with cols[0]:
            st.metric("Sparsity", f"{stats['sparsity']:.3f}")
        with cols[1]:
            st.metric("Active Neurons", stats['active_neurons'])
        with cols[2]:
            st.metric("Dead Neurons", 
                     self.sae.encoder.weight.shape[0] - stats['active_neurons'])
        
        # Feature importance
        st.subheader("Feature Importance")
        weights = self.sae.encoder.weight.data
        importance = torch.norm(weights, dim=1)
        
        fig = px.bar(
            x=range(len(importance)),
            y=importance.detach().numpy(),
            title="Neuron Importance Distribution",
            labels={"x": "Neuron", "y": "Importance"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Token attribution
        if self.tokens is not None:
            st.subheader("Token Attribution")
            attribution = torch.matmul(self.activations, weights.T)
            
            fig = px.imshow(
                attribution.detach().numpy(),
                x=self.tokens,
                title="Token-Feature Attribution",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistical analysis
        st.subheader("Statistical Measures")
        with st.expander("View Details"):
            stats_df = pd.DataFrame({
                "Mean": self.activations.mean(dim=0).detach().numpy(),
                "Std": self.activations.std(dim=0).detach().numpy(),
                "Max": self.activations.max(dim=0)[0].detach().numpy(),
                "Min": self.activations.min(dim=0)[0].detach().numpy()
            })
            st.dataframe(stats_df)