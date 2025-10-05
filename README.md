# Project overview

This repo contains a from-scratch LLM implementation emphasizing clarity and maintainability. The model uses a transformer backbone and a Mixture-of-Experts (MoE) layer (or layers) to increase model capacity while keeping compute reasonable. Clean code, typed interfaces, and modular components let you experiment with expert counts, gating strategies, and routing constraints easily.

# Key features

* Minimal, readable transformer implementation (attention, feed-forward, residuals, layernorm).

* Modular MoE block(s) with pluggable gating strategies (softmax gating, top-k gating).

* Expert implementations are simple MLP blocks but easily replaceable with more complex specialists (e.g., convolutional experts, adapters).

* Load balancing losses and capacity constraints included.

* Training scripts with mixed-precision and distributed training hooks.

* Inference-time efficient routing and optional expert pruning.

# Future Scope
Planning to include Reinforcement Learning with human feedback (RLHF)
