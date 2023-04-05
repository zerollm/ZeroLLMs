"""
A single file implementation of a GPT Language Model.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
3) nanoGPT PyTorch implementation
https://github.com/karpathy/nanoGPT/blob/master/model.py
4) Attention Is All You Need
https://arxiv.org/abs/1706.03762
5) Language Models are Unsupervised Multitask Learners
https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

n_layers, d_model, n_heads, d_head, n_vocab, n_ctx
"""
from typing import Optional, Union
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    n_ctx: int = 1024
    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_vocab: int = 50304
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    dropout: float = 0.0
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    bias: bool = True
    # should make sure tokenizer and model use the same pad_token_id
    pad_token_id: int = 0
    # `ignore_index` does not contribute to the input gradient.
    ignore_index: int = -100


def gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class FFN(nn.Module):
    """ a fully connected feed-forward network
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        d_ff = config.d_model * 4
        self.c_fc = nn.Linear(config.d_model, d_ff, bias=config.bias)
        self.c_proj = nn.Linear(d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SelfMultiHeadAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_heads
        self.dropout = config.dropout

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.n_ctx, config.n_ctx)).view(1, 1, config.n_ctx, config.n_ctx))

    def forward(self, x):
        B, T, D = x.size()  # batch size, sequence length, embedding dimension (d_model)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh: n_heads, dh: d_head
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh)

        # causal self-attention; Self-attend: (B, nh, T, dh) x (B, nh, dh, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, dh) -> (B, nh, T, dh)

        y = y.transpose(1, 2).contiguous().view(B, T, D)

        # output projection
        y = self.c_proj(y)
        y = self.residual_dropout(y)
        return y


class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = SelfMultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.ffn = FFN(config)

    def forward(self, x):
        # Layer normalization (Ba et al., 2016) was moved to the input of each sub-block.
        x = self.ln_1(x)
        x = x + self.attn(x)
        # an additional layer normalization was added after the final self-attention block.
        x = self.ln_2(x)
        x = x + self.ffn(x)
        return x


def tie_or_clone_weights(output_embeddings: nn.Module, input_embeddings: nn.Module, torchscript=False):
    """ Make sure we are sharing the input and output embeddings.
        Export to TorchScript can't handle parameter sharing so we are cloning them instead.
    """
    if torchscript:
        output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
    else:
        output_embeddings.weight = input_embeddings.weight

    if getattr(output_embeddings, "bias", None) is not None:
        output_embeddings.bias.data = nn.functional.pad(
            output_embeddings.bias.data,
            (
                0,
                output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
            ),
            "constant",
            0,
        )


class GPT(nn.Module):

    reserved_configs = {
        'gpt2':         GPTConfig(n_layers=12, n_heads=12, d_model=768),   # 124M params
        'gpt2-medium':  GPTConfig(n_layers=24, n_heads=16, d_model=1024),  # 350M params
        'gpt2-large':   GPTConfig(n_layers=36, n_heads=20, d_model=1280),  # 774M params
        'gpt2-xl':      GPTConfig(n_layers=48, n_heads=25, d_model=1600),  # 1558M params
    }

    def __init__(self, config: Union[GPTConfig, str]):
        super().__init__()
        if isinstance(config, str):
            config = self.reserved_configs[config]
        self.config = config
        self.pad_token_id = config.pad_token_id
        self.ignore_index = config.ignore_index

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.config.n_vocab, self.config.d_model),
                wpe=nn.Embedding(self.config.n_ctx, self.config.d_model),
                drop=nn.Dropout(self.config.dropout),
                blocks=nn.ModuleList([Block(self.config) for _ in range(self.config.n_layers)]),
                ln_f=LayerNorm(self.config.d_model, bias=self.config.bias),
            ))
        self.lm_head = nn.Linear(config.d_model, config.n_vocab, bias=False)
        # https://paperswithcode.com/method/weight-tying
        tie_or_clone_weights(self.lm_head, self.transformer.wte)

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # apply special scaled init to the residual projections, per GPT-2 paper
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in self.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(0.02 / math.sqrt(2 * self.config.n_layers)))

    def num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None):
        b, t = input_ids.size()
        assert t <= self.config.n_ctx, f"Cannot forward sequence of length {t}, the max context length = {self.config.n_ctx}"

        device = input_ids.device
        position = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        # element of input_ids should in [0 ~ n_vocab - 1], negative index is not supported for nn.Embedding
        tok_emb = self.transformer.wte(input_ids)  # token embeddings of shape (b, t, d_model)
        pos_emb = self.transformer.wpe(position)  # position embeddings of shape (1, t, d_model)
        x = tok_emb + pos_emb
        x = self.transformer.drop(x)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)

        loss = None
        if labels is not None:
            labels[labels == self.pad_token_id] = self.ignore_index  # All labels set to -100 are ignored (masked)
            lm_logits = self.lm_head(x)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens and ignore_index default is -100
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.ignore_index)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            x = x[:, [-1], :]
            lm_logits = self.lm_head(x)  # note: using list [-1] to preserve the time dim

        return lm_logits, loss

    @ torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int, temperature=1.0, top_k: Optional[int] = None):
        """
        Generates sequences of token ids for models with a language modeling head. only for decoder models.
        input_ids: (bsz, t)
        """
        bsz, t = input_ids.size()
        assert t <= self.config.n_ctx, f"the input_ids is out of the model max context {self.config.n_ctx}"
        assert temperature <= 1 and temperature >= 0, "temperature should be in 0 ~ 1"
        if top_k is not None:
            assert top_k > 0 and top_k < self.config.n_vocab, f"invalid top_k: {top_k}"

        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_ids)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, top_k_ids = torch.topk(logits, dim=-1, k=top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)  # (bsz, n_vocab)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (bsz, 1)
            # append sampled index to the running sequence and continue
            input_ids = torch.cat((input_ids, idx_next), dim=1)

        return input_ids

    @staticmethod
    def from_pretrained_hf(model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        pass

    @staticmethod
    def export_to_hf():
        pass
