class CharEncoder:
    def __init__(self, train_on):
        self.found_tokens = list(set(train_on))

        lookup = dict()
        for i, char_token in enumerate(self.found_tokens):
            lookup[char_token] = i

        self.tokens = torch.empty((len(train_on),), dtype=torch.long)
        for i, char in enumerate(train_on):
            self.tokens[i] = lookup[char]

    @property
    def n_vocab(self):
        return len(self.found_tokens)

    def decode(self, indices):
        return "".join([self.found_tokens[i] for i in indices])

