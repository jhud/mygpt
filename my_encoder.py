import tiktoken
import torch

class MyEncoder:
    def __init__(self, train_on):
        self.enc = tiktoken.get_encoding("p50k_base")
        encoded = self.enc.encode(train_on)
        self.found_tokens = list(set(encoded))
        lookup = dict()
        for i, p50_token in enumerate(self.found_tokens):
            lookup[p50_token] = i

        self.tokens = torch.empty((len(encoded),), dtype=torch.long)
        for i, p50 in enumerate(encoded):
            self.tokens[i] = lookup[p50]

    @property
    def n_vocab(self):
        return len(self.found_tokens)

    def decode(self, indices):
        return self.enc.decode([self.found_tokens[i] for i in indices])
