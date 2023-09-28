# mygpt
An easily-trained baby GPT that can stand in for the real thing. Based on Andrej Karpathy's makemore, but set up to mimic a llama-cpp server.

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

