import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

def main():
    # Setup directories
    dataset_path = "./Dataset/"
    output_dir = "./Feature/esm2/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all protein IDs and sequences
    datasets = ["Train_335.pkl", "Test_60.pkl", "Test_315-28.pkl", "UBtest_31-6.pkl"]
    protein_sequences = {}
    
    for ds_name in datasets:
        ds_path = os.path.join(dataset_path, ds_name)
        if not os.path.exists(ds_path):
            print(f"Warning: {ds_path} not found.")
            continue
        with open(ds_path, "rb") as f:
            data = pickle.load(f)
        for pid, val in data.items():
            # val is [sequence, label]
            protein_sequences[pid] = val[0]
            
    print(f"Loaded {len(protein_sequences)} unique protein IDs from datasets.")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = "facebook/esm2_t33_650M_UR50D"
    print(f"Loading tokenizer and model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    
    # Use fp16 on GPU to save memory and prevent OOM
    if device.type == "cuda":
        model = model.half()
    model = model.to(device)
    model.eval()
    
    # Generate embeddings
    for pid, seq in tqdm(protein_sequences.items(), desc="Generating ESM-2 embeddings"):
        out_path = os.path.join(output_dir, f"{pid}.npy")
        if os.path.exists(out_path):
            continue
            
        # Try running on GPU first, fallback to CPU if OOM occurs
        try:
            with torch.no_grad():
                inputs = tokenizer(seq, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)
                
                # Forward pass
                outputs = model(input_ids)
                # Extract representations
                # inputs shape: (1, L+2)
                # outputs.last_hidden_state shape: (1, L+2, 1280)
                # We extract indices 1 to L+1
                seq_len = len(seq)
                embedding = outputs.last_hidden_state[0, 1:seq_len+1, :].cpu().numpy()
                
                # Check shape correctness
                assert embedding.shape == (seq_len, 1280), f"Shape mismatch for {pid}: {embedding.shape} vs ({seq_len}, 1280)"
                
                np.save(out_path, embedding.astype(np.float16)) # Save as float16 to conserve space
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == "cuda":
                print(f"\nCUDA OOM for protein {pid} of length {len(seq)}. Falling back to CPU...")
                torch.cuda.empty_cache()
                
                # Fallback to CPU
                try:
                    # Temporarily move model to CPU
                    model = model.to("cpu").float() # Move model to CPU and convert back to float32
                    with torch.no_grad():
                        inputs = tokenizer(seq, return_tensors="pt")
                        input_ids = inputs["input_ids"].to("cpu")
                        outputs = model(input_ids)
                        seq_len = len(seq)
                        embedding = outputs.last_hidden_state[0, 1:seq_len+1, :].cpu().numpy()
                        assert embedding.shape == (seq_len, 1280)
                        np.save(out_path, embedding.astype(np.float16))
                except Exception as ex:
                    print(f"Error generating embedding for {pid} on CPU: {ex}")
                finally:
                    # Move model back to GPU and fp16
                    model = model.half().to(device)
            else:
                print(f"\nError for {pid}: {e}")
                raise e

    print("Embedding generation completed successfully!")

if __name__ == "__main__":
    main()
