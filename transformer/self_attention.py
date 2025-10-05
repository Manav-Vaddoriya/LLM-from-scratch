import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def manual_self_attention_example():
    """
    Tiny example showing self-attention computation step-by-step
    Sequence length = 3, embedding dimension = 4
    """
    print("=" * 60)
    print("MANUAL SELF-ATTENTION COMPUTATION")
    print("=" * 60)
    
    # Input: 3 tokens, each with 4-dim embedding
    X = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],  # Token 1
        [0.0, 2.0, 0.0, 2.0],  # Token 2
        [1.0, 1.0, 1.0, 1.0]   # Token 3
    ])
    print(f"\nInput X (shape {X.shape}):\n{X}")
    
    d_k = 4  # Key dimension
    
    # Weight matrices (simplified - usually these are learned)
    W_q = torch.eye(4)  # Query weights
    W_k = torch.eye(4)  # Key weights
    W_v = torch.eye(4)  # Value weights
    
    # Step 1: Compute Q, K, V
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    print(f"\nQueries Q:\n{Q}")
    print(f"\nKeys K:\n{K}")
    print(f"\nValues V:\n{V}")
    
    # Step 2: Compute attention scores (Q @ K^T)
    scores = Q @ K.T
    print(f"\nAttention Scores (Q @ K^T):\n{scores}")
    
    # Step 3: Scale by sqrt(d_k)
    scores_scaled = scores / math.sqrt(d_k)
    print(f"\nScaled Scores (/ sqrt({d_k})):\n{scores_scaled}")
    
    # Step 4: Apply softmax
    attention_weights = F.softmax(scores_scaled, dim=-1)
    print(f"\nAttention Weights (after softmax):\n{attention_weights}")
    print(f"(Each row sums to 1: {attention_weights.sum(dim=-1)})")
    
    # Step 5: Weighted sum of values
    output = attention_weights @ V
    print(f"\nOutput (Attention_Weights @ V):\n{output}")
    print(f"Output shape: {output.shape}")
    
    return output


# ============================================================================
# 1.3 SINGLE ATTENTION HEAD
# ============================================================================

class SingleAttentionHead(nn.Module):
    """Single attention head implementation"""
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
    
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]
        Q = self.W_q(x)  # [batch_size, seq_len, d_k]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [batch_size, seq_len, seq_len]
        
        # Apply mask if provided (for causal/padding masks)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights