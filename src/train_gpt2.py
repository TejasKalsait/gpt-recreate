from dataclasses import dataclass

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

import tiktoken

import os
import time
import inspect
import sys

# Parameters
relative_input_path = '../data/shakespear.txt'

total_batch_size = 524288       # in terms of tokens(B, T) 2**19
#total_batch_size = 524288 // 4

micro_batch_size = 4
micro_block_size = 512

max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_epochs = 10

epochs = 50

# Class to store the model parameters
@dataclass
class GPTConfig:
    batch_size : int = 16
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
        # flag to indicate this layer should be scaled to maintain std 1.0 after residual connection
        self.c_proj.RESIDUAL_SCALE = 1
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
        
        
        # Attention of tokens calculation using traditional method. We are used to this and
        # torch.compile does not call fuse attention on this
        #
        # attention : torch.Tensor = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))    # [B, nh, T, hs] x [B, nh, hs, T]  --> [B, nh, T, T]
        # # Masking the attention since we are working on a decoder and also applying the softmax
        # attention = attention.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # attention = F.softmax(attention, dim = -1)
        # y : torch.Tensor = attention @ v        # [B, nh, T, T] x [B, nh, T, hs]  --> [B, nh, T, hs]

        # Implementing Fuse attention block instead
        y = F.scaled_dot_product_attention(q, k, v, is_causal = True)

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
        # flag to indicate this layer should be scaled to maintain std 1.0 after residual connection
        self.c_proj.RESIDUAL_SCALE = 1

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

        # weight sharing scheme that saves computation (recommended in paper)
        self.transformer.wte.weight = self.lm_head.weight

        # Apply some function by passing each nn.Module one by one
        self.apply(self._init_weight)

    # Weight initializer
    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'RESIDUAL_SCALE'):
                std = (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, x, y = None):

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

        # Calculating loss
        loss = None
        if y is not None:
            # cross entropy expect x shape [B, T] and y shape [B*T,]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)) ,y.view(-1))
        
        return logits, loss
    
    def configure_optimizer(self, weight_decay, learning_rate, device):

        # Collecting all the parameters that require gradients
        param_dict = {pn : p for pn, p in self.named_parameters()}
        param_dict = {pn : p for pn, p in param_dict.items() if p.requires_grad}

        # All 2D and above parameters will require weight decay and other 1D parameters like
        # Layer norm, GELU, won't so seperate them out
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]

        # Two seperate groups of parameters
        optim_params = [
            {'params': decay_params, 'weight_decay' : weight_decay},
            {'params': nondecay_params, 'weight_decay' : 0.0}
        ]

        # Parameter Stats
        num_decay_param = sum(p.nelement() for p in decay_params)
        num_nondecay_param = sum(p.nelement() for p in nondecay_params)
        if master_process:
            print(f"Total Decay parameters --> {num_decay_param}")
            print(f"Total Non-Decay parameters --> {num_nondecay_param}")

        # Checking if the fused kernal option is available, if yes we want to make an optimizer that uses it
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"Using fused AdamW --> {use_fused}")
            print("_________________________________________________________________________")

        # Initialize optimizer and return
        optimizer = torch.optim.AdamW(optim_params, lr = learning_rate, betas = (0.9, 0.95), eps = 1e-8, fused = use_fused)
        return optimizer

# Creating a data loader that retuns a fresh batch of x, y
class DataLoader:

    def __init__(self, B, T, input_file, process_rank, num_processes) -> None:
        # saving shapes
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        # Open file and save data at initialization
        with open(input_file, 'r') as f:
            text = f.read()
        # encode text and create tokens
        encoder = tiktoken.get_encoding("gpt2")
        tokens = encoder.encode(text)
        self.tokens = torch.tensor(tokens)
        if master_process:
            # stats
            print(f"Loaded data. Total {len(self.tokens)} tokens available")
            print(f"1 epoch consists of {len(self.tokens) // (B * T)} major batches")

        # From which index to sample the batch from (depends on which process this is)
        self._start = self.B * self.T * self.process_rank

    def next_batch(self):

        B, T = self.B, self.T   # batch size and context size
        # index from startin point to start + batch*tokens + 1
        # +1 because we need to save that as y's last value
        buffer = self.tokens[self._start:(B * T) + self._start + 1]
        # Build x and y 
        x = buffer[:-1].view(B, T)      # [B, T]
        y = buffer[1:].view(B, T)       # [B, T]
        # Move the start to next batch start
        self._start += (B * T * self.num_processes)
        # In case next batch sampling will go out of bounds, reset to zero
        if self._start + (B * T + 1) > len(self.tokens):
            self._start = self.B * self.T * self.process_rank

        return x, y
# ________________________________________________________________________________________________________________

# Cosine decay learning rate scheduler
def get_lr(epoch):
    
    # Linear warmup going on
    if epoch < warmup_epochs:
        return max_lr * (epoch+1) / warmup_epochs
    # If epoch > total epochs (always return min possible then)
    if epoch > epochs:
        return min_lr
    # In between, cosine decay
    decay_ratio = (epoch - warmup_epochs) / (epochs - warmup_epochs)
    assert 0 <= decay_ratio <= 1,0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# Set up DDP (Distributed Data Parallel)
ddp = int(os.environ.get('RANK', -1)) != -1     # Bool if ddp is present
if ddp:
    # Setup ddp
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])                  # Unique ID for each process of each node
    ddp_local_rank = int(os.environ['LOCAL_RANK'])      # Unique ID for each process within a node (Used for setting GPU number)
    ddp_world_size = int(os.environ['WORLD_SIZE'])      # Number of processes participating
    device = f'cuda:{ddp_local_rank}'                   # Setting which GPU
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0                      # One master process for prints
else:
    # vanilla
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # Setting device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

# Autocast device type should be 'cuda' and not 'cuda:0'
autocast_device = 'cuda'

# print training on what device
if master_process:
    print("Training on -->", device)

# File Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file_path = os.path.join(script_dir, relative_input_path)

# Setting the precision for speed to take advantage of Nvidia TF32
torch.set_float32_matmul_precision('high')  # Default highest

# Setting seeds
torch.manual_seed(1010)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1010)

# Gradient Accumulation to simulate bigger batch sizes
assert total_batch_size % (micro_batch_size*micro_block_size*ddp_world_size) == 0, f"The total block size {total_batch_size} is not divisible by micro_batch_size * micro_block_size * num_processes {micro_batch_size}*{micro_block_size}*{ddp_world_size}"
# Accumulate gradients for these many steps
grad_accum_steps = total_batch_size // (micro_batch_size*micro_block_size*ddp_world_size)
if master_process:          # Only printed by master process
    print(f"{grad_accum_steps} micro batches will be processed in one major batch")

# Initializing DataLoader, Model, Optimizer
data_loader = DataLoader(micro_batch_size, micro_block_size, input_file_path, process_rank = ddp_rank, num_processes = ddp_world_size)
model = GPT(GPTConfig(vocab_size = 50304))
model.to(device)
model = torch.compile(model)
# This container wraps the model in a DDP class which makes no change in the forward pass, but
# during backward pass, when the gradients from each process is accumulated, it averages out all gradients
# from each process and populates the average in each process's gradients
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])
# Unwrapping the raw model from the DDP wrap to pass to optimizer
raw_model = model.module if ddp else model
optimizer = model.configure_optimizer(weight_decay = 0.1, learning_rate = 6e-4, device = device)


# _____________________________________________________________________________________________________

# Training Loop
for epoch in range(epochs):
    # record time
    tock = time.time()

    # Zero grad
    optimizer.zero_grad()

    # To keep track of loss for the batch instead of micro_batch
    loss_accum = 0.0

    # Gradient accumulation for simulating bigger batches
    for micro_step in range(grad_accum_steps):
        # Fetch a new batch
        x, y = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # Feed forward with dtype = torch.bfloat16 whenever possible (some operations) to speed up matmul
        # with torch.autocast(device_type = device, dtype = torch.bfloat16):
        with torch.autocast(device_type = autocast_device, dtype = torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps      # To normalize the mean-like loss which is lost when acccumulating (Check NOTES)
        loss_accum += loss.detach()
        # Disable sync and only enable it on the last step again so that it syncs gradients only on last step of micro bacth loop
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # Zero grad and backward pass
        loss.backward()

    # Enabling sync of losses across all the processes for the master process to print 
    if ddp:
        dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG)

    # Clipping the gradient to avoid shock inscase a bad batch comes in.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Get learning rate based on the learning rate scheduler and use this in optimizer
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Update step
    optimizer.step()

    # To have cuda and cpu sync with no pending works
    torch.cuda.synchronize()

    # record time
    tick = time.time()
    dt = (tick - tock)

    # How many tokens processed per sec
    tokens_processed = data_loader.B * data_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    # Print Stats
    if master_process:
        print(f"Epoch {epoch}/{epochs} | Loss --> {loss_accum:.2f}| Grad norm --> {norm:4f} | LR --> {lr:.4f} | Time --> {dt:.2f} | Tokens/sec --> {tokens_per_sec:.2f}")

# Clean exit of processes
if ddp:
    destroy_process_group()

sys.exit(0)

# _____________________________ GENERATE ___________________________________________________________

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