# mygpt
An easily-trained baby GPT that can stand in for the real thing. Based on Andrej Karpathy's makemore, but set up to mimic a llama-cpp server.

The main points of differentiation are:
 - my version is token-based (tiktoken)
 - code to load up multiple text files as a training set
 - a minimal server which is a drop-in replacement for the OpenAI REST API
 - extra inference parameters, such as top_k, and the supression of tokens which you do not want to see (ie glitch tokens or annoyingly repeated tokens).



So you can train the default tiny 15M parameter model, and use that in your projects instead of ChatGPT.


This is not production-ready; it's a toy implementation for educational purposes.

## Setup

pip install -r requirements.txt

Add as many text files as you want to the "data" folder as a trianing set.

## Using it

It is not very configurable at the moment -tweak the code in main.py to get it to do what you want.

### Training

Uncomment "train()" in main.py. It will save checkpoints of the model parameters into the "models" folder.

### Inference / text generation

Once you have trained the model, comment "train()" and uncomment "inference()". Setup whatever prompt you want. Then run the script to see the generated text appear.



## Example output

These are some sample responses from a model trained on a dozen old Encyclopedia Brittanica volumes for a couple of hours on an NVidia 4GB GPU, then fine-tuned on 120 dad-jokes from the internet.

```
Q: What is a dog?
A: To get a frog.

Q: Why did the chicken cross the road?
A: Because it was Sunday.
```

With "Q: " as a prompt, it will make its own "jokes":

```
Q: How do you cross a race with no cold birds?
A: Because they did the toothache entirely.

Q: Why did a figureur hit a like?
A: Because a joke.
````

Pure comic genius!

The prompt format is:
```
Q: {user question}
```

Here is part of the fine-tuning set (real dad jokes from the internet - not what was generated):
Q: What do you call a fake noodle? A: An impasta
Q: How do you organise a space party? A: You planet!


## Improving the performance

This code is an educational exercise; it has a self-attention head built from first principles.

You can get a great memory + speed saving by replacing the self-attention head with a prebuilt module from pytorch. Replace the old Block class with this code:

```
class Block(nn.Module):
    """ Intersperse communication (attention) and computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = nn.MultiheadAttention(n_embd, n_head, dropout=dropout)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        l1_results = self.ln1(x)
        sa_results, _ = self.sa(l1_results, l1_results, l1_results, need_weights=False)
        x = x + sa_results  # note the x+ is a residual connection to help with optimisation
        x = x + self.ffwd(self.ln2(x))
        return x
```

I got a ~30% memory reduction.

This is probably because many separate steps in our layers can be mathematically simplified when the layers are combined into a single module. 
Plus, pytorch is no doubt much better optimised than our naive python implementation.

