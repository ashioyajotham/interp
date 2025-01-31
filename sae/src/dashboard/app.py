import streamlit as st
import torch
import numpy as np
import plotly.express as px

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.model_loader import ModelLoader
from src.models.autoencoder import SparseAutoencoder
from src.evaluation.analyzer import SAEAnalyzer
from components.sidebar import render_sidebar
from components.viz import plot_activations, plot_features, plot_sparsity_dist

def initialize_page():
    st.set_page_config(layout="wide", page_title="SAE Interpretability")
    st.title("Sparse Autoencoder Interpretability")

def main():
    initialize_page()
    configs = render_sidebar()
    
    st.header("Text Analysis")
    text = st.text_area("Input Text", "The quick brown fox jumps over the lazy dog")
    
    if st.button("Analyze", type="primary"):
        with st.spinner("Running analysis..."):
            model = ModelLoader(configs["model"])
            activations = model.get_activations(text, configs["layer"])
            
            # Ensure proper shape for visualization
            if len(activations.shape) == 3:
                activations_2d = activations.squeeze(0)  # (1, seq, dim) -> (seq, dim)
            else:
                activations_2d = activations
            
            sae = SparseAutoencoder(
                input_dim=activations_2d.shape[1],
                hidden_dim=configs["hidden_dims"],
                sparsity_param=configs["sparsity"]
            )
            
            tab1, tab2, tab3 = st.tabs(["Activations", "Features", "Analysis"])
            
            with tab1:
                st.plotly_chart(
                    plot_activations(activations, configs["threshold"]),
                    use_container_width=True
                )
            
            with tab2:
                st.plotly_chart(
                    plot_features(sae.encoder.weight.data),
                    use_container_width=True
                )
                
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(
                        plot_sparsity_dist(activations_2d),
                        use_container_width=True
                    )
                with col2:
                    analyzer = SAEAnalyzer(sae)
                    stats = analyzer.analyze_sparsity(activations_2d)
                    st.json(stats)

if __name__ == "__main__":
    main()
