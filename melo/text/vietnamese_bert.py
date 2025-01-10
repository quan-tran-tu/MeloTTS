import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Optional
import sys

# Use PhoBERT as the pretrained model
model_id = 'vinai/phobert-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = None

def get_bert_feature(
    text: str,
    word2ph: List[int],
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Extract BERT features for Vietnamese text.
    
    Args:
        text: Input text to process
        word2ph: List of integers representing word-to-phoneme mapping
        device: Device to run the model on ('cuda', 'cpu', or 'mps')
        
    Returns:
        torch.Tensor: Phoneme-level features (shape: [hidden_size, num_phonemes])
    """
    global model
    
    # Handle device selection
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    
    # Initialize model if not already done
    if model is None:
        model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
        model.eval()  # Set to evaluation mode
    
    # Extract features
    with torch.no_grad():
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt")
        
        # Move inputs to appropriate device
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        
        # Get model outputs
        outputs = model(**inputs, output_hidden_states=True)
        
        # Get hidden states from the last 3 layers
        hidden_states = outputs.hidden_states
        last_hidden_states = torch.cat(hidden_states[-3:-2], -1)[0].cpu()
        
        # Verify the shapes match
        assert inputs["input_ids"].shape[-1] == len(word2ph), \
            f"Mismatch between BERT tokens ({inputs['input_ids'].shape[-1]}) and word2ph ({len(word2ph)})"
        
        # Create phone-level features
        phone_level_feature = []
        for i in range(len(word2ph)):
            # Repeat the word-level feature for each phone
            repeat_feature = last_hidden_states[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        
        # Concatenate all phone-level features
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
    
    return phone_level_feature.T

if __name__ == "__main__":
    # Test the BERT feature extraction
    test_text = "Xin chào thế giới!"
    test_word2ph = [1, 1, 1, 1, 1]  # Example word2ph mapping
    
    # Extract features
    features = get_bert_feature(test_text, test_word2ph, device="cpu")
    print("Feature shape:", features.shape)
    print("Feature device:", features.device)
    print("Feature type:", features.dtype)