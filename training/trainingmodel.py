import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import time
import sys
import os
import matplotlib.pyplot as plt
import math

# Add the project's root directory to the Python path.
project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(project_root)
sys.path.append(project_root)

from transformer_block import TransformerBlock, SinusoidalPositionalEmbedding
from training.dataset import DatasetClass 


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, num_blocks, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = SinusoidalPositionalEmbedding(d_model, max_seq_len)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_blocks)]
        )
        self.layer_norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, targets=None):
        seq_len = input_ids.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len).to(input_ids.device)
        tok_emb = self.token_embedding(input_ids)
        x = self.positional_embedding(tok_emb)
        for block in self.transformer_blocks:
            x, _ = block(x, mask)
        x = self.layer_norm_final(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50):
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx_cond = input_ids if input_ids.size(1) <= model.max_seq_len else input_ids[:, -model.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token_id), dim=1)
    output_text = tokenizer.decode(input_ids[0].tolist())
    model.train()
    return output_text

@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, loss = model(inputs, targets)
        total_loss += loss.item()
    model.train()
    return total_loss / len(data_loader)

if __name__ == "__main__":
    BATCH_SIZE = 4          
    MAX_SEQ_LEN = 256       
    STRIDE = 256
    D_MODEL = 768           
    NUM_BLOCKS = 12        
    NUM_HEADS = 12          
    D_FF = D_MODEL * 4
    DROPOUT = 0.1
    NUM_EPOCHS = 10
    BASE_LEARNING_RATE = 1e-4
    WARMUP_STEPS = 10      
    
    # --- Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    text_file_path = os.path.join(project_root, 'THE VERDICT.txt')
    try:
        with open(text_file_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Dataset 'THE VERDICT.txt' loaded with {len(text):,} characters.")
    except FileNotFoundError:
        print(f"ERROR: '{text_file_path}' not found. Please make sure the file is in your project root.")
        sys.exit(1) # Exit if the file doesn't exist
    
    n = len(text)
    train_text = text[:int(n*0.9)]
    val_text = text[int(n*0.9):]

    tokenizer = tiktoken.get_encoding('gpt2')
    VOCAB_SIZE = tokenizer.n_vocab 

    train_dataset = DatasetClass(train_text, tokenizer, MAX_SEQ_LEN, STRIDE)
    val_dataset = DatasetClass(val_text, tokenizer, MAX_SEQ_LEN, STRIDE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False)
    
    model = GPTLanguageModel(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN, NUM_BLOCKS, NUM_HEADS, D_FF, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LEARNING_RATE)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
    
    if len(train_loader) == 0:
        print("\nCRITICAL ERROR: The training dataset is too small to create even one batch.")
        print("Please use a larger dataset or decrease MAX_SEQ_LEN.")
        sys.exit(1)
        
    print(f"Training on {len(train_loader)} batches, validating on {len(val_loader)} batches.")

    train_losses = []
    val_losses = []
    lr_history = []
    
    # --- Training Loop ---
    step = 0
    total_steps = len(train_loader) * NUM_EPOCHS
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        epoch_train_loss = 0
        
        for inputs, targets in train_loader:
            # Learning Rate Scheduler Logic
            if step < WARMUP_STEPS:
                lr = BASE_LEARNING_RATE * (step + 1) / WARMUP_STEPS
            else:
                progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
                lr = BASE_LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            lr_history.append(lr)

            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            epoch_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        avg_val_loss = float('inf')
        if len(val_loader) > 0:
            avg_val_loss = evaluate(model, val_loader, device)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
    print("\nTraining complete!")
    
    if len(val_loader) > 0:
        print(f"\nFinal Test Loss (on validation set): {val_losses[-1]:.4f}")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(train_losses, label='Training Loss')
    if all(l != float('inf') for l in val_losses):
        ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Losses')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(lr_history, label='Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Generate Sample Text ---
    print("\n--- Final Generated Sample ---")
    prompt = "The evening had been a triumph"
    generated_text = generate(model, tokenizer, prompt, max_new_tokens=100)
    print(f"PROMPT: '{prompt}'")
    print(f"GENERATION: '{generated_text}'\n")
    # --- Hyperparameters ---
    BATCH_SIZE = 16
    MAX_SEQ_LEN = 128
    STRIDE = 128
    D_MODEL = 256
    NUM_BLOCKS = 4
    NUM_HEADS = 12
    D_FF = D_MODEL * 4
    DROPOUT = 0.1
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5
    
    # --- Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    text_file_path = os.path.join(project_root, 'THE VERDICT.txt')
    with open(text_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # --- Split Data (90% train, 5% validation, 5% test) ---
    n = len(text)
    train_text = text[:int(n*0.9)]
    val_text = text[int(n*0.9):int(n*0.95)]
    test_text = text[int(n*0.95):]

    # --- Create Tokenizer and DataLoaders ---
    tokenizer = tiktoken.get_encoding('gpt2')
    VOCAB_SIZE = tokenizer.n_vocab

    train_dataset = DatasetClass(train_text, tokenizer, MAX_SEQ_LEN, STRIDE)
    val_dataset = DatasetClass(val_text, tokenizer, MAX_SEQ_LEN, STRIDE)
    test_dataset = DatasetClass(test_text, tokenizer, MAX_SEQ_LEN, STRIDE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False)
    
    # --- Model and Optimizer ---
    model = GPTLanguageModel(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN, NUM_BLOCKS, NUM_HEADS, D_FF, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
    # Add a check to prevent running if a dataloader is empty
    if len(val_loader) == 0:
        print("\nWARNING: Validation data is too small for the batch size. Consider using a larger validation split or a larger text file.")
    else:
        print(f"Training on {len(train_loader)} batches, validating on {len(val_loader)} batches.")

    # --- Training Loop ---
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        epoch_train_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            epoch_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Calculate average losses for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Only evaluate on validation if the loader is not empty
        avg_val_loss = float('inf')
        if len(val_loader) > 0:
            avg_val_loss = evaluate(model, val_loader, device)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
    print("\nTraining complete!")

    # --- Final Evaluation on Test Set ---
    if len(test_loader) > 0:
        test_loss = evaluate(model, test_loader, device)
        print(f"\nFinal Test Loss: {test_loss:.4f}")


    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    if all(l != float('inf') for l in val_losses):
        plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
