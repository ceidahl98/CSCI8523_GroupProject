import torch
import torchvision
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as f
import math

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


# From Andrej Karpathy GPT-2
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()


        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d1 = nn.Dropout(config.dropout)



    def forward(self, x):
        B, T, C = x.size()  # batch, tokens (seq length), classes (embedding dimension)
        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B,n_head,T,head size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # att = (q @ k.transpose(-2,-1)) * (1.0/ math.sqrt(k.size(-1)))

        # att = att.masked_fill(self.mask[:,:,16,:T] == 0, float('-inf'))  #make sure attention can't see future tokens

        # att = f.softmax(att,dim=-1)
        # y = att @ v



        y = f.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = self.d1(y)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.d1 = nn.Dropout(config.dropout)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.INIT_SCALE = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.d1(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 4
    vocab_size: int = 2048
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1



class GPT(nn.Module):

    def __init__(self, config, in_size):
        super().__init__()
        self.config = config
        self.d1 = nn.Dropout(config.dropout)
        self.activation = nn.SiLU()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Linear(in_size,config.n_embd),  # vocab index to vocab embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),  # input sequence to positional embedding
            late = nn.Embedding(720,config.n_embd),
            lote = nn.Embedding(1440,config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # hidden layers consisting of transformer blocks
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weights = self.lm_head.weight  # output linear layer is the same as token embedding

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = .02
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=.02)

    def forward(self, indicies,lats,lons):

        B, T, E = indicies.shape

        lats=lats.unsqueeze(1).repeat((1,T))
        lons=lons.unsqueeze(1).repeat((1,T))

        assert T <= self.config.block_size, f"seq_length greater than block_size"

        pos = torch.arange(0, T, dtype=torch.long, device=indicies.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(indicies)
        lat_emb = self.transformer.late(lats)
        lon_emb = self.transformer.lote(lons)

        tok_emb = self.activation(tok_emb)

        # act_emb = self.transformer.ate(actions)

        x = tok_emb
        # x = self.d1(x)

        x = x + pos_emb + lat_emb + lon_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return logits


