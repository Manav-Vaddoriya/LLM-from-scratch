import tiktoken
from typing import List

class TiktokenWrapper:
    def __init__(self, model_name: str):
        """
        Initializes the tokenizer by loading a tiktoken encoding for a specific model.
        Args:
            model_name (str): The name of the model to get the encoding for 
                              (e.g., "gpt-4", "gpt-3.5-turbo").
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
            self.model_name = model_name
            print(f"Tiktoken encoding for '{self.model_name}' loaded successfully.")
            print(f"Encoding name: '{self.encoding.name}'")
        except KeyError:
            print(f"Model '{model_name}' not found. Please use a valid model name like 'gpt-4' or 'gpt-3.5-turbo'.")
            self.encoding = None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            self.encoding = None

    def encode(self, text: str) -> List[int]:
        """
        Encodes a string of text into a list of token IDs.
        Args:
            text (str): The input string to encode.

        Returns:
            List[int]: A list of integer token IDs.
        """
        if not self.encoding:
            print("Encoding is not loaded. Cannot encode.")
            return []
        return self.encoding.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        Args:
            token_ids (List[int]): The list of token IDs to decode.
        Returns:
            str: The decoded, human-readable string.
        """
        if not self.encoding:
            print("Encoding is not loaded. Cannot decode.")
            return ""
        return self.encoding.decode(token_ids)
        
    def get_tokens(self, text: str) -> List[bytes]:
        """
        Gets the raw byte representations of tokens for a given text.
        Args:
            text (str): The input string.
        Returns:
            List[bytes]: A list of the token byte sequences.
        """
        if not self.encoding:
            print("Encoding is not loaded. Cannot get tokens.")
            return []
        
        encoded_ids = self.encoding.encode(text)
        return [self.encoding.decode_single_token_bytes(token_id) for token_id in encoded_ids]

if __name__ == "__main__":
    tokenizer = TiktokenWrapper(model_name="gpt-4")
    
    if tokenizer.encoding:
        sample_text = "tiktoken is a fast and efficient BPE tokenizer."

        print("\n" + "="*40)
        print(f"Original Text:\n'{sample_text}'")
        print("-" * 40)

        encoded_ids = tokenizer.encode(sample_text)
        print(f"Encoded Token IDs:\n{encoded_ids}")
        print("-" * 40)
        
        tokens = tokenizer.get_tokens(sample_text)
        print(f"Tokens (as bytes):\n{tokens}")
        print("-" * 40)

        decoded_text = tokenizer.decode(encoded_ids)
        print(f"Decoded Text:\n'{decoded_text}'")
        print("="*40)

        assert sample_text == decoded_text
        print("Verification successful: Original and decoded texts match!")