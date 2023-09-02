# This model is largely based on Andrej Karparthy's "makemore" videos, with some refinements to the inference method.

import torch
import torch.nn as nn
from torch.nn import functional as F

# for victorian
# device = "cuda" if torch.cuda.is_available() else "cpu"
# batch_size = 64 # How many sequences in parallel - 64 saturates my 4GB CUDA card pretty well
# block_size = 32 # context length, was 8
# n_embd = 120 # num embedding dimensions
# n_layer = 16
# dropout = 0.2

# for encyclopedias
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 36 # How many sequences in parallel - 64 saturates my 4GB CUDA card pretty well
block_size = 64 # context length, was 8
n_embd = 128 # num embedding dimensions
n_layer = 20
dropout = 0.3


def get_batch(tokens):
    ix = torch.randint(len(tokens) - block_size, (batch_size, ))
    x = torch.stack([tokens[i:i+block_size] for i in ix])
    y = torch.stack([tokens[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# K = "here's what I am"
# Q = "here's what I'm looking for in my past"
# V = "here's what I will output based on what I found"

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ Simple computation layer so we are not raw-dogging the output of the attention heads. """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Intersperse communication (attention) and computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,
                                     head_size)  # all smaller heads are concatenated to give same size output as embedding size
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # note the x+ is a residual connection to help with optimisation
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        n_heads = 4
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_heads) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)  # (B. T. vocab size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # Put tensor in form that cross_entropy func accepts
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, stop_tokens=None, generated=None, top_k=5, sample=True, supress_tokens=[]):
        """ Generate a stream of tokens.
          :param idx prompt
          :param max_new_tokens max tokens to generate
          :param stop_tokens immediately stop on any of these tokens
          :param generated callback for when a new token is generated. Useful for chatbot behaviour.
          :param top_k only sample the top n most likely tokens.
          :param sample always choose the top most likely token if false, otherwise take based on probability.
          :param supress_tokens choose tokens to supress, ie glitch tokens or anything that messes with your output. Try removing 'and' or 'the'.
          :return the generated tokens."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  #
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            if top_k:
                topk_values, topk_indices = torch.topk(logits, top_k)
                threshold = topk_values[:, [-1]]
                logits[logits < threshold] = float('-inf')

            #print("B:"+str(logits[:,930].item()))

            if len(supress_tokens) > 0:
                logits = logits.index_fill(1, supress_tokens, float('-inf'))

            #print("A:"+str(logits[:,930].item()))

            probs = F.softmax(logits, dim=-1)
            if sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            if generated:
                generated(idx_next)
            idx = torch.cat((idx, idx_next), dim=1)
            next_token = idx_next.item()
            if stop_tokens and next_token in stop_tokens:
                return idx
        return idx

