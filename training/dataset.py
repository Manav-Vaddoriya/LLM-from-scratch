import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class DatasetClass(Dataset):
    def __init__(self, text: str, tokenizer, maxlength: int, stride: int):
        self.inp_ids = []
        self.tar_ids = []

        '''Tokenize the entire text corpus once.
        This is more efficient than tokenizing line by line.'''
        token_ids = tokenizer.encode(text)

        '''Use a sliding window to create chunks of text.
        the window moves by `stride` tokens at each step.'''
        for i in range(0, len(token_ids) - maxlength, stride):

            # Input chunk is the sequence of `maxlength`
            input_chunk = token_ids[i : i + maxlength]
            
            '''Target chunk is the input chunk shifted by one position to the right.
            This sets up the next-token prediction task.'''
            output_chunk = token_ids[i + 1 : i + maxlength + 1]

            self.inp_ids.append(torch.tensor(input_chunk))
            self.tar_ids.append(torch.tensor(output_chunk))

    def __len__(self):
        return len(self.inp_ids)

    def __getitem__(self, idx):
        return self.inp_ids[idx], self.tar_ids[idx]

def create_dataloader_v1(text: str, batch_size: int = 4, maxlength: int = 256, 
                         stride: int = 128, shuffle: bool = True, drop_last: bool = True, 
                         num_workers: int = 0):
    """
    Factory function to create a DataLoader for the GPTDatasetV1.
    Args:
        text (str): The raw text data to process.
        batch_size (int): Number of samples per batch.
        maxlength (int): The context window size or sequence length.
        stride (int): The number of tokens to slide the window forward.
        shuffle (bool): Whether to shuffle the data.
        drop_last (bool): Whether to drop the last incomplete batch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        Tuple[DataLoader, tiktoken.Encoding]: A tuple containing the configured 
                                             DataLoader and the tokenizer instance.
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')

    # Create the Dataset (Corrected class name to GPTDatasetV1)
    dataset = DatasetClass(text, tokenizer, maxlength, stride)

    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader, tokenizer

if __name__ == '__main__':
    # Sample text corpus
    with open("THE VERDICT.txt", "r", encoding="utf-8") as f:
        full_text = f.read()
    
    # Create the dataloader using the factory function
    train_loader = create_dataloader_v1(full_text, batch_size=8, maxlength=128, stride=64)

    print(f"Created a DataLoader with {len(train_loader)} batches.")

    # Iterate over one batch to see the output shape
    for inputs, targets in train_loader:
        print("\n--- Sample Batch ---")
        print(f"Inputs shape: {inputs.shape}")  
        print(f"Targets shape: {targets.shape}")
        print(f"\nSample input from batch:\n{inputs[0]}")
        print(f"\nSample target from batch:\n{targets[0]}")
        break