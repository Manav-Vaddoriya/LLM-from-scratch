import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tiktoken
import matplotlib.pyplot as plt
import time
import math
from training.dataset import DatasetClass 
from transformer_modi import TransformerBlock, SinusoidalPositionalEmbedding


class GPTLanguageModel(nn.Module):
    """The full GPT model, updated to build with MoE blocks."""
    def __init__(self, vocab_size, d_model, max_seq_len, num_blocks, num_heads, d_ff, num_experts, top_k, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = SinusoidalPositionalEmbedding(d_model, max_seq_len)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, num_experts, top_k, dropout) for _ in range(num_blocks)]
        )
        self.layer_norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, targets=None):
        seq_len = input_ids.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).view(1, 1, seq_len, seq_len)
        tok_emb = self.token_embedding(input_ids)
        x = self.positional_embedding(tok_emb)
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = self.layer_norm_final(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Calculates average loss on a given dataset."""
    model.eval()
    total_loss = 0
    if len(data_loader) == 0:
        return float('inf')
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, loss = model(inputs, targets)
        total_loss += loss.item()
    model.train()
    return total_loss / len(data_loader)

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100):
    """Generates new text from a prompt."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx_cond = input_ids if input_ids.size(1) <= model.max_seq_len else input_ids[:, -model.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token_id), dim=1)
    output_text = tokenizer.decode(input_ids[0].tolist())
    model.train()
    return output_text


if __name__ == "__main__":
    # --- Hyperparameters ---
    BATCH_SIZE = 4
    MAX_SEQ_LEN = 256
    STRIDE = 128
    D_MODEL = 768
    NUM_BLOCKS = 12
    NUM_HEADS = 12
    D_FF = D_MODEL * 4
    DROPOUT = 0.1
    NUM_EPOCHS = 10
    BASE_LEARNING_RATE = 1e-4
    WARMUP_STEPS = 10
    NUM_EXPERTS = 8
    TOP_K = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    text_file_path = os.path.join(project_root, 'THE VERDICT.txt')
    with open(text_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    n = len(text)
    train_text = text[:int(n*0.8)]
    val_text = text[int(n*0.8):int(n*0.9)]
    test_text = text[int(n*0.9):]

    tokenizer = tiktoken.get_encoding('gpt2')
    VOCAB_SIZE = tokenizer.n_vocab

    train_dataset = DatasetClass(train_text, tokenizer, MAX_SEQ_LEN, STRIDE)
    val_dataset = DatasetClass(val_text, tokenizer, MAX_SEQ_LEN, STRIDE)
    test_dataset = DatasetClass(test_text, tokenizer, MAX_SEQ_LEN, STRIDE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False)
    
    # --- Model and Optimizer ---
    model = GPTLanguageModel(
        VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN, NUM_BLOCKS, 
        NUM_HEADS, D_FF, NUM_EXPERTS, TOP_K, DROPOUT
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LEARNING_RATE)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
    print(f"Using a Mixture of {NUM_EXPERTS} Experts, selecting top {TOP_K} per token.")
    
    # --- Training Loop & Plotting ---
    train_losses, val_losses, lr_history = [], [], []
    step = 0
    total_steps = len(train_loader) * NUM_EPOCHS if len(train_loader) > 0 else 0

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        epoch_train_loss = 0
        
        if len(train_loader) == 0:
            print("Warning: Training data loader is empty. Skipping epoch.")
            continue

        for inputs, targets in train_loader:
            # Learning Rate Scheduler Logic
            if total_steps > WARMUP_STEPS: 
                if step < WARMUP_STEPS:
                    lr = BASE_LEARNING_RATE * (step + 1) / WARMUP_STEPS
                else:
                    progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
                    lr = BASE_LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))
            else:
                lr = BASE_LEARNING_RATE

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
        avg_val_loss = evaluate(model, val_loader, device)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
    print("\nTraining complete!")
    test_loss = evaluate(model, test_loader, device)
    print(f"Final Test Loss: {test_loss:.4f}")

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
