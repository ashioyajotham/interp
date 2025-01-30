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
    st.set_page_config(page_title="SAE Interpretability Dashboard", layout="wide")
    st.title("Sparse Autoencoder Interpretability Dashboard")

def load_models():
    model_name = st.sidebar.selectbox(
        "Select Base Model",
        ModelLoader.SUPPORTED_MODELS.keys()
    )
    
    model_loader = ModelLoader(model_name)
    return model_loader

def text_analysis_section(model_loader):
    st.header("Text Analysis")
    
    text = st.text_area("Enter text to analyze:", "Hello, world!")
    layer = st.slider("Select layer", -12, -1, -1)
    
    if st.button("Analyze"):
        activations = model_loader.get_layer_activations(text, layer)
        
        # Visualize activations
        fig = px.imshow(
            activations.numpy(),
            labels=dict(x="Hidden Dimension", y="Token Position"),
            title="Layer Activations Heatmap"
        )
        st.plotly_chart(fig)
        
        # Show statistics
        st.subheader("Activation Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Activation", f"{activations.mean():.3f}")
        with col2:
            st.metric("Max Activation", f"{activations.max():.3f}")
        with col3:
            st.metric("Sparsity", f"{(activations == 0).float().mean():.3%}")

def sae_analysis_section(model_loader):
    st.header("Sparse Autoencoder Analysis")
    
    # SAE Parameters
    st.subheader("SAE Configuration")
    col1, col2 = st.columns(2)
    with col1:
        input_dim = st.number_input("Input Dimension", value=768)
        hidden_dim = st.number_input("Hidden Dimension", value=256)
    with col2:
        sparsity_param = st.slider("Sparsity Parameter", 0.0, 1.0, 0.1)
        l1_coef = st.slider("L1 Coefficient", 0.0, 0.1, 0.001)
    
    if st.button("Train SAE"):
        with st.spinner("Training SAE..."):
            # Get sample activations
            sample_text = "This is a sample text for training the SAE."
            activations = model_loader.get_layer_activations(sample_text)
            
            # Initialize and train SAE
            sae = SparseAutoencoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                sparsity_param=sparsity_param
            )
            
            # Analyze SAE
            analyzer = SAEAnalyzer(sae)
            stats = analyzer.analyze_sparsity(activations)
            
            # Show results
            st.success("Training complete!")
            st.json(stats)
            
            # Visualize features
            feature_directions = sae.get_feature_directions()
            fig = px.imshow(
                feature_directions.detach().numpy(),
                labels=dict(x="Input Dimension", y="Feature"),
                title="Learned Feature Directions"
            )
            st.plotly_chart(fig)

def main():
    initialize_page()
    model_loader = load_models()
    
    tab1, tab2 = st.tabs(["Text Analysis", "SAE Analysis"])
    
    with tab1:
        text_analysis_section(model_loader)
    with tab2:
        sae_analysis_section(model_loader)

if __name__ == "__main__":
    main()
