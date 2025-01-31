import streamlit as st

def render_sidebar():
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