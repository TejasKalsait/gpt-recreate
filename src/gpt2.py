from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# Class to store the model parameters
@dataclass
class GPTConfig:
    block_size : int= 1024      # Context length
    vocab_size : int = 50257    # 50000 merges, 256 binary, 1 |<endoftext>|
    n_layer : int = 12
    n_head : int = 12
    n_embed : int = 768

# Self attention to gather and share information across tokens
class CasualSelfAttention(nn.Module):

    def __init__(self, config : GPTConfig):
        super().__init__()

        assert config.n_embed % config.n_head == 0, f"Number of heads -> {config.n_head} should be perfectly divisible by number of embeddings -> {config.n_embed}" 
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)     # Concatenated key, query, value
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)         # projection because of skip connection
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", mask)

    def forward(self, x : torch.Tensor):
        # B-> Batch T-> Tokens(time) C-> Channels(embeddings)
        B, T, C = x.size()
        # Passing everything as linear layers and then splitting it
        kqv =  self.c_attn(x)        # [B, T, 3C]
        q, k, v = kqv.split(self.n_embed, dim = 2)      # Each key, query and value have shape [B, T, C]
        # We view each vector in the shape of [B, nh, T, hs]
        k = k.contiguous().view(B, T, self.n_head, C // self.n_head).transpose(1, 2)        # All these have shapes [B, num_head, T, head_size]
        q = k.contiguous().view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = k.contiguous().view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Attention of tokens calculation
        attention : torch.Tensor = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))    # [B, nh, T, hs] x [B, nh, hs, T]  --> [B, nh, T, T]
        # Masking the attention since we are working on a decoder and also applying the softmax
        attention = attention.masked_fill(self.bias[:, :, T, T] == 0, float('-inf'))
        attention = F.softmax(attention, dim = -1)
        y : torch.Tensor = attention @ v        # [B, nh, T, T] x [B, nh, T, hs]  --> [B, nh, T, hs]
        # y is of shape [B, nh, T, hs] and we need it again in [B, T, C] form
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Projection layer for skip connection shape match
        y = self.c_proj(y)

        return y

# Multi layer perceptron to do computation after self attention
class MLP(nn.Module):

    def __init__(self, config : GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# Attention block of CasualSelfAttention (comminication) and MLP (computation)
class Block(nn.Module):
    def __init__(self, config : GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):

        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# GPT class stores the model
class GPT(nn.Module):

    def __init__(self, config : GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias = False)

    def forward(self, x):

        # Input data size
        B, T = x.size()
        # we need to make sure that T is not more than the block size, it can be less tho
        assert T <= self.config.block_size, f"The sequence of tokens cannot be more than {self.config.block_size} you passed in {T} tokens"

        # Embedding the pos + tok
        pos = torch.arange(0, T, dtype=torch.long, device = x.device)   # [T]
        pos_emb = self.transformer.wpe(pos)                             # [T, n_embed]
        tok_emb = self.transformer.wte(x)                               # [B, T, n_embed]
        x = pos_emb + tok_emb                                           # [T, n_embed] + [B, T, n_embed] --> [B, T, n_embed]
        # Forwarding to hidden blocks of the transformer
        for block in self.transformer.h:                                # [B, T, C] -> [B, T, C]
            x = block(x)
        # Forwarding the final layer
        x = self.transformer.ln_f(x)                                    # [B, T, C]
        logits = self.lm_head(x)                                        # [B, T, vocab_size]
        
        return logits
    

# ________________________________________________________________________________________________________________

# Hyperparameters
num_returned_sequences = 5
max_length = 75

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Training on -->", device)

model = GPT(GPTConfig())
model = model.to(device)    # Switching to GPU

# Creating tokens
encoder = tiktoken.get_encoding("gpt2")
tokens = encoder.encode("Hi. Let me tell you a story about the planet Pluto. Once,")
tokens = torch.tensor(tokens, dtype = torch.long)   # [T,]
tokens = tokens.unsqueeze(0).repeat(num_returned_sequences, 1)  # [B, T]
x = tokens.to(device)

# Generating Loop
torch.manual_seed(55)
torch.cuda.manual_seed(55)

while x.size(1) < max_length:
    with torch.no_grad():
        # Feedforward
        logits = model(x)
        # Grab the last token logits
        logits = logits[:, -1, :]       # [B, vocab_size]
        # Get the probabilities to sample from
        probs = F.softmax(logits, dim = -1)     # [B, vocab_size]
        # Only consider the top 50 probs by default, Take the top 50 probs and make others zero and resample it.
        # This way we never choose next token that is less likely even by chance. Everything is from the top 50
        top_probs, top_indices = probs.topk(k = 50, dim = -1)       # [B, 50]
        # Now sample from this top_prob (returns the index of chosen value from dist)
        ix = torch.multinomial(top_probs, num_samples = 1)       # [B, 1]
        # Gather the topk incides
        xval = torch.gather(top_indices, -1, ix)        # [B, 1]
        # Append to the input sequence to repeat
        x = torch.cat([x, xval], dim=-1)                # [B, 1]


# Printing the generated text
for i in range(num_returned_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = encoder.decode(tokens)
    print(">", decoded)