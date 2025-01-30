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
    text = st.text_area("Enter text:", "The quick brown fox jumps over the lazy dog", key="text_input")
    
    if st.button("Process Text", key="process_btn"):
        try:
            activations = model_loader.get_activations(text)
            st.success("Text processed successfully!")
            return activations
        except Exception as e:
            st.error(f"Error processing text: {str(e)}")
    return None

def sae_analysis_section(model_loader):
    st.header("Sparse Autoencoder Analysis")
    
    # Get activations first
    activations = text_analysis_section(model_loader)
    
    if activations is not None:
        if st.button("Run SAE Analysis", key="analyze_btn"):
            try:
                # Initialize SAE
                sae = SparseAutoencoder(
                    input_dim=activations.shape[1],
                    hidden_dim=64
                )
                
                # Analyze
                analyzer = SAEAnalyzer(sae)
                stats = analyzer.analyze_sparsity(activations)
                
                # Display results
                st.json(stats)
                
                # Feature visualization
                feature_dirs = sae.encoder.weight.data.T
                fig = px.imshow(
                    feature_dirs.detach().numpy(),
                    title="Feature Directions"
                )
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

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
