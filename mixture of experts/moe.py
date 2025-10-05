import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """A simple feed-forward network expert."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    """A sparse Mixture of Experts layer."""
    def __init__(self, d_model, d_ff, num_experts, top_k, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, d_ff, dropout) for _ in range(num_experts)])
        self.gating_network = nn.Linear(d_model, num_experts)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        num_tokens = x_flat.shape[0]

        # Gating: Get routing decisions from the gating network
        logits = self.gating_network(x_flat)
        
        # Top-k Selection: Find the top_k experts and their weights for each token
        gates, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(gates, dim=-1)

        # Sparse Computation: Combine expert outputs
        final_output = torch.zeros_like(x_flat)
        
        # Get a flat list of token indices
        token_indices = torch.arange(num_tokens, device=x.device)

        for i in range(self.top_k):
            expert_indices = indices[:, i]
            expert_weights = weights[:, i].unsqueeze(-1)
            
            for exp_idx in range(self.num_experts):
                # Find all tokens that are routed to this expert for this top_k choice
                mask = (expert_indices == exp_idx)
                if mask.any():
                    # Get the specific tokens for this expert
                    tokens_for_expert = x_flat[mask]
                    
                    # Process them with the expert
                    expert_output = self.experts[exp_idx](tokens_for_expert)
                    
                    # Multiply by the gate weight
                    weighted_output = expert_output * expert_weights[mask]
                    
                    # Add the weighted output to the correct position in the final output tensor
                    final_output.index_add_(0, token_indices[mask], weighted_output)

        return final_output.view(batch_size, seq_len, d_model)