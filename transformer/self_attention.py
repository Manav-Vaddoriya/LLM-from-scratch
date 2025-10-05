import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def manual_self_attention_example():
    X = torch.tensor([
        [1.0, 0.0, 1.0, 0.0], 
        [0.0, 2.0, 0.0, 2.0],  
        [1.0, 1.0, 1.0, 1.0]   
    ])
    print(f"\nInput X (shape {X.shape}):\n{X}")
    
    d_k = 4  
    
    W_q = torch.eye(4)  
    W_k = torch.eye(4) 
    W_v = torch.eye(4) 
    
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    print(f"\nQueries Q:\n{Q}")
    print(f"\nKeys K:\n{K}")
    print(f"\nValues V:\n{V}")
    
    scores = Q @ K.T
    print(f"\nAttention Scores (Q @ K^T):\n{scores}")
    
    scores_scaled = scores / math.sqrt(d_k)
    print(f"\nScaled Scores (/ sqrt({d_k})):\n{scores_scaled}")
    
    attention_weights = F.softmax(scores_scaled, dim=-1)
    print(f"\nAttention Weights (after softmax):\n{attention_weights}")
    print(f"(Each row sums to 1: {attention_weights.sum(dim=-1)})")
    
    output = attention_weights @ V
    print(f"\nOutput (Attention_Weights @ V):\n{output}")
    print(f"Output shape: {output.shape}")
    
    return output

class SingleAttentionHead(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
    
    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
