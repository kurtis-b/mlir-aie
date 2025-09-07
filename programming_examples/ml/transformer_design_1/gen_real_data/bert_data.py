import torch
from transformers import BertModel, BertTokenizer
import os
import numpy as np

def save_bert_weights(model_name='bert-base-uncased', output_dir='bert_weights'):
    os.makedirs(output_dir, exist_ok=True)
    model = BertModel.from_pretrained(model_name)
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        # Save each weight as a .npy file for easy loading in C++ (e.g., with Eigen or OpenCV)
        npy_path = os.path.join(output_dir, f"{name}.npy")
        torch_np = param.cpu().numpy()
        # Save as float32 for compatibility
        torch_np = torch_np.astype('float32')
        np.save(npy_path, torch_np)

def generate_bert_input_output_cases(model_name='bert-base-uncased', output_dir='bert_inp_out', texts=None, max_length=256):
    os.makedirs(output_dir, exist_ok=True)

    if texts is None:
        # Generate dummy texts of increasing length
        base_text = "Hello world. "
        texts = [base_text * l for l in range(1, max_length+1, 32)]  # e.g., lengths: 1, 33, 65, ..., up to 257 tokens

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    for idx, text in enumerate(texts):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            token_embeddings = model.embeddings.word_embeddings(input_ids)
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeddings = model.embeddings.position_embeddings(position_ids)
            token_type_embeddings = model.embeddings.token_type_embeddings(token_type_ids)
            embeddings = token_embeddings + position_embeddings + token_type_embeddings
            embeddings = model.embeddings.LayerNorm(embeddings)
            embeddings = model.embeddings.dropout(embeddings)

            # Convert attention_mask to float for compatibility with encoder
            attention_mask = attention_mask.to(dtype=embeddings.dtype)

            outputs = model.encoder(
                embeddings,
                attention_mask=attention_mask,
                head_mask=[None] * model.config.num_hidden_layers,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            last_hidden_state = outputs.last_hidden_state

        # Save input embeddings and output for each case
        np.save(os.path.join(output_dir, f"input_embeddings_{idx}.npy"), embeddings.cpu().numpy().astype("float32"))
        np.save(os.path.join(output_dir, f"output_hidden_state_{idx}.npy"), last_hidden_state.cpu().numpy().astype("float32"))
   

if __name__ == "__main__":
    save_bert_weights()
    texts = ["Hello world. This is a test.", "Another example text for BERT."]
    generate_bert_input_output_cases(texts=texts)
    
    with open("sparsity.txt", "w") as f:
        # Calculate sparsity for bert_weights
        for fname in os.listdir('bert_weights'):
            if fname.endswith('.npy'):
                arr = np.load(os.path.join('bert_weights', fname))
                sparsity = float(np.count_nonzero(arr == 0)) / arr.size
                f.write(f"{fname}: {sparsity:.6f}\n")
        # Calculate sparsity for bert_inp_out
        for fname in os.listdir('bert_inp_out'):
            if fname.endswith('.npy'):
                arr = np.load(os.path.join('bert_inp_out', fname))
                sparsity = float(np.count_nonzero(arr == 0)) / arr.size
                f.write(f"{fname}: {sparsity:.6f}\n")