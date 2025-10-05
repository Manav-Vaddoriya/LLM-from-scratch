from dataclasses import dataclass
@dataclass
class Config:
    # Data
    corpus_path: str = "data/corpus.txt"  
    bpe_merges: int = 10000
    vocab_size: int = 30000
    max_seq_len: int = 256
    stride: int = 256
    
    # Model
    d_model: int = 768
    num_heads: int = 12
    d_ff: int = d_model * 4
    num_blocks: int = 12
    dropout: float = 0.1
    
    # Training
    batch_size: int = 4
    num_epochs: int = 10
    base_learning_rate: float = 1e-4
    warmup_steps: int = 10
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    
    # Checkpoint
    save_path: str = "checkpoints/"
    save_every: int = 1

config = Config()
