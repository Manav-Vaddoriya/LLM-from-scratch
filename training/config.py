from dataclasses import dataclass


@dataclass
class Config:
# Data / tokenizer
    corpus_path: str = "data/corpus.txt" # plain text corpus
    bpe_merges: int = 10000
    vocab_size: int = 30000
    max_seq_len: int = 256


    # Model
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 6
    dropout: float = 0.1


    # Training
    batch_size: int = 32
    epochs: int = 10
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"


    # Checkpoint
    save_path: str = "checkpoints/"
    save_every: int = 1


config = Config()