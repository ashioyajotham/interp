from transformers import AutoModel, AutoTokenizer
import torch

class ModelLoader:
    SUPPORTED_MODELS = {
        'gpt2': 'gpt2',
        'gpt2-medium': 'gpt2-medium',
        'bert-base': 'bert-base-uncased',
        'roberta-base': 'roberta-base'
    }

    def __init__(self, model_name: str):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_name = self.SUPPORTED_MODELS[model_name]
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    def get_layer_activations(self, text: str, layer_idx: int = -1) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get activations from specified layer
        hidden_states = outputs.hidden_states[layer_idx]
        return hidden_states.squeeze(0)  # Remove batch dimension
