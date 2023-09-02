# A FastAPI server which mimics the python-cpp server.
# Only implements barely enough to be usable.

import asyncio
import json

import torch
import tiktoken

from model import GPTModel, get_batch, device

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Sequence
from contextlib import asynccontextmanager
from sse_starlette.sse import EventSourceResponse

enc = tiktoken.get_encoding("p50k_base")
m = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global enc, m
    # Load the ML model
    model = GPTModel(enc.n_vocab)
    m = model.to(device)
    print("Loading model...")
    m.load_state_dict(torch.load("models/encyclopedias.pt"))

    yield
    # Clean up the ML models and release the resources

app = FastAPI(lifespan=lifespan)


class CreateCompletionRequest(BaseModel):
    prompt: str
    stream: bool = False
    stop: Sequence[str]
    max_tokens: int = 100
    temperature: float

@app.get("/v1/models")
async def models(request: Request):
    return {"data": [{"id": "mygpt"}]}

@app.post("/v1/completions")
async def completions(request: Request, body: CreateCompletionRequest):
    global enc, m

    def on_generated(token):
        return

    stops = [{"choices": [{"text": enc.encode(stop)[0]}]} for stop in body.stop]

    prompt_tokens = enc.encode(body.prompt)
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    ret = m.generate(idx, max_new_tokens=body.max_tokens, stop_tokens=stops, generated=on_generated)

    tokens = ret.tolist()[0]

    if body.stream:
        chunks = [json.dumps({"choices": [{"text": enc.decode([token])}]}) for token in tokens]

        def new_messages():
            # Check if data in table
            if len(chunks) == 0:
                return None
            else:
                return True

        async def event_generator():
            while True:
                if await request.is_disconnected():
                    break

                if new_messages():
                    yield chunks.pop(0)
                else:
                    break

                await asyncio.sleep(0.01)

        return EventSourceResponse(event_generator())
    else:
        return {"resp": enc.decode(tokens)}

def text_from_path(path):
    import glob
    text = ""
    for filename in glob.glob(path+"*.txt"):
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                text += line
    return text

def text_from_file():
    text = ""
    with open('data/input.txt') as f:
        lines = f.readlines()
        for line in lines:
            text += line
    return text


def train():
    global enc, m
    text = text_from_path("data/encyclopedias/")

    tokens = torch.tensor(enc.encode(text), dtype=torch.long)

    print(f"Vocab size {enc.n_vocab}")
    print(f"Training data size {len(tokens)}")

    model = GPTModel(enc.n_vocab)
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    try:
        m.load_state_dict(torch.load("models/encyclopedias.pt"))
    except FileNotFoundError:
        print("Model file not found, starting new training...")
    print("Loaded model. Beginning training...")

    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    epoch = 300
    for checkpoints in range(30000):
        loss_total = 0
        for steps in range(epoch):
            xb, yb = get_batch(tokens)
            logits, loss = m(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        print(loss_total/epoch)

        torch.save(model.state_dict(), "models/encyclopedias.pt")
    print("Done with training!")


def inference():
    global m, enc

    model = GPTModel(enc.n_vocab)
    m = model.to(device)

    print("Loading model...")
    m.load_state_dict(torch.load("models/encyclopedias.pt"))

    def on_generated(token):
        #print(f"{enc.decode([token])}({token.item()})", end="")
        print(enc.decode([token]), end="")


    stops = [enc.encode(".")[0], enc.encode("?")[0], enc.encode("!")[0]]
    prompt = "Wollongong is a large town in"
    print(f"{prompt}", end="")
    prompt_tokens = enc.encode(prompt)
    supress = torch.tensor([0, 930], dtype=torch.long, device=device) # the and and: 290, 262
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    m.generate(idx, max_new_tokens=400, stop_tokens=None, generated=on_generated, top_k=3, sample=True, supress_tokens=supress)
    print("\n")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(f"Using device {device}")

    train()
    #inference()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
