import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    """Multi-head attention with splitting, concatenation, and final projection"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Single linear layers for all heads (more efficient)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x):
        """Split last dimension into (num_heads, d_k)"""
        batch_size, seq_len, d_model = x.size()
        # Reshape: [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k]
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose: [batch, num_heads, seq_len, d_k]
        return x.transpose(1, 2)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections for all heads at once
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split into multiple heads
        Q = self.split_heads(Q)  # [batch, num_heads, seq_len, d_k]
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [batch, num_heads, seq_len, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: [batch, num_heads, seq_len, d_k]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2)  # [batch, seq_len, num_heads, d_k]
        attn_output = attn_output.contiguous().view(batch_size, -1, self.d_model)
        # attn_output: [batch, seq_len, d_model]
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output, attn_weights