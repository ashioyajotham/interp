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

def initialize_page():
    st.set_page_config(layout="wide", page_title="SAE Interpretability")
    st.title("Sparse Autoencoder Interpretability")

def setup_sidebar():
    with st.sidebar:
        st.header("Model Configuration")
        model_name = st.selectbox(
            "Select Model", 
            ["bert", "gpt2"], 
            key="model_select"
        )
        st.divider()
        return model_name

def main():
    initialize_page()
    model_name = setup_sidebar()
    
    # Initialize or get model from session state
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = ModelLoader(model_name)
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Input Analysis")
        text = st.text_area(
            "Enter text to analyze:",
            value="The quick brown fox jumps over the lazy dog",
            height=100,
            key="input_text"
        )
        
        if text:
            activations = st.session_state.model_loader.get_activations(text)
            activations = activations.view(-1, activations.size(-1))
            
            # Initialize SAE
            sae = SparseAutoencoder(
                input_dim=activations.shape[1],
                hidden_dim=64
            )
            
            # Analyze
            analyzer = SAEAnalyzer(sae)
            stats = analyzer.analyze_sparsity(activations)
            
            with st.expander("Activation Statistics", expanded=True):
                st.json(stats)
    
    with col2:
        st.subheader("Feature Visualization")
        if 'activations' in st.session_state:
            fig = px.imshow(
                activations.detach().numpy(),
                title="Neuron Activations"
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
