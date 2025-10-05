import torch
import torch.nn as nn
from transformer.positional_encoding import LearnedPositionalEmbedding, SinusoidalPositionalEmbedding
from transformer.self_attention import manual_self_attention_example
from transformer.multihead_att import MultiHeadAttention
from transformer.ffn import FeedForwardNetwork
from transformer.layernormalization import LayerNorm
from transformer.self_attention import SingleAttentionHead


class TransformerBlock(nn.Module):
    """Complete Transformer encoder block with all components"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Layer normalization (2 instances)
        self.norm1 = nn.LayerNorm(d_model) 
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection and layer norm
        attn_output, attn_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # Residual + LayerNorm
        
        # Feed-forward with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  # Residual + LayerNorm
        
        return x, attn_weights

def demonstrate_components():
    """Demonstrate all components with example usage"""
    
    # Hyperparameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    
    print("\n" + "=" * 60)
    print("DEMONSTRATING TRANSFORMER COMPONENTS")
    print("=" * 60)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")
    
    # Positional Embeddings
    print("\n1. POSITIONAL EMBEDDINGS")
    print("-" * 40)
    
    learned_pe = LearnedPositionalEmbedding(max_seq_len=100, d_model=d_model)
    x_with_learned_pe = learned_pe(x)
    print(f"With learned PE: {x_with_learned_pe.shape}")
    
    sinusoidal_pe = SinusoidalPositionalEmbedding(d_model=d_model)
    x_with_sin_pe = sinusoidal_pe(x)
    print(f"With sinusoidal PE: {x_with_sin_pe.shape}")
    
    # Manual Self-Attention
    print("\n2. MANUAL SELF-ATTENTION")
    print("-" * 40)
    manual_self_attention_example()
    
    # Single Attention Head
    print("\n3. SINGLE ATTENTION HEAD")
    print("-" * 40)
    single_head = SingleAttentionHead(d_model=d_model, d_k=64)
    output, weights = single_head(x)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Multi-Head Attention
    print("\n4. MULTI-HEAD ATTENTION")
    print("-" * 40)
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output, weights = mha(x)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Number of heads: {num_heads}, d_k per head: {d_model // num_heads}")
    
    # Feed-Forward Network
    print("\n5. FEED-FORWARD NETWORK")
    print("-" * 40)
    ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff)
    output = ffn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden dimension (d_ff): {d_ff} (expansion factor: {d_ff/d_model}x)")
    
    # Full Transformer Block
    print("\n6. FULL TRANSFORMER BLOCK")
    print("-" * 40)
    transformer_block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1
    )
    output, weights = transformer_block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in transformer_block.parameters())
    print(f"Total parameters in block: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_components()