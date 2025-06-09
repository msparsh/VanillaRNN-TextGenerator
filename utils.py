import numpy as np
import tensorflow as tf

class Vectorizer:
    def __init__(self, text):
        self.text = text
        # Build vocabulary from the unique characters in the text
        self.vocab = sorted(set(text))
        self.vocab_size = len(self.vocab)
        # Create mappings: char -> index and index -> char
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

    def tokenize(self, text=None):
        # Convert the text into a list of token indices processing one character at a time
        if not text: 
            text = self.text
        tokens = [self.char2idx[char] for char in text]
        return tokens

    def vectorize(self, tokens, add_batch_dim=True):
        # Convert token list into one-hot encoded vectors.
        # By default returns a 3D tensor of shape (1, sequence_length, vocab_size)
        one_hot = tf.one_hot(tokens, depth=self.vocab_size)
        if add_batch_dim:
            one_hot = tf.expand_dims(one_hot, axis=0)
        return one_hot

    def detokenize(self, token_sequence):
        # Convert a sequence of token indices back to a text stringZ
        text = ''.join([self.idx2char[token] for token in token_sequence])
        return text


# Example usage (for testing purposes)
if __name__ == "__main__":
    sample_text = "Hello World"
    vectorizer = Vectorizer(sample_text)
    
    tokens = vectorizer.tokenize()
    print("Tokens:", tokens)
    
    one_hot_vectors = vectorizer.vectorize(tokens)
    # Now one_hot_vectors.shape should be (1, len(text), vocab_size)
    print("One-hot vectors shape:", one_hot_vectors.shape)
    
    # To detokenize using the original 1D token list:
    reconstructed_text = vectorizer.detokenize(tokens)
    print("Reconstructed Text:", reconstructed_text)