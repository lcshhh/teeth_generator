from typing import Dict, List, Optional, Tuple, Callable
import logging
import torch
import os
import argparse
import torch.nn as nn
import torch.nn.functional as F
# from model.Dit import DiT
import math
from models import *
import yaml
from einops import rearrange
from utils.builder import MODELS 
from models.pointnet_plus_encoder import StyleEncoder, ShapeEncoder

class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t,
                                     2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-
                              1).reshape(*xyz.shape[:-1], -1)

        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb

class MLP(torch.nn.Module):
    """MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        dtype = torch.float32
        self.dense_h_to_4h = nn.Linear(
            hidden_size,
            hidden_size * 4,
            bias=False,
            dtype=dtype,
        )

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        self.dense_4h_to_h = nn.Linear(
            hidden_size * 2,
            hidden_size,
            bias=False,
            dtype=dtype,
        )

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(seq_len, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
        
        xq_ = torch.view_as_complex(xq_)
        xk_ = torch.view_as_complex(xk_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
        return xq_out.type_as(xq), xk_out.type_as(xk)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.freqs_cis = precompute_freqs_cis(embed_dim,32)
        self.mlp = MLP(embed_dim)

    def forward(self, x, condition_embedding=None, key_padding_mask=None, attn_mask=None):

        x = self.ln_1(x)
        # a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        q,k = apply_rotary_emb(x,x,self.freqs_cis.to(x.device))
        if condition_embedding is not None:
            q += condition_embedding
            k += condition_embedding
        a, _ = self.attn(q, k, x, key_padding_mask=key_padding_mask, need_weights=False, attn_mask=attn_mask)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x

@MODELS.register_module()
class StyleTransformer(nn.Module):
    def __init__(self,
        config
        ):
        super(StyleTransformer, self).__init__()
        self.encoders = nn.ModuleList([StyleEncoder(config) for _ in range(32)])
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(Block(config.z_dim, config.num_heads))
        self.pos_embed = PositionEmbeddingCoordsSine(3, config.z_dim, 10000)
        self.shape_encoder = ShapeEncoder(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.z_dim, config.z_dim),
            nn.LeakyReLU(),
            nn.Linear(config.z_dim, config.z_dim),
            nn.LeakyReLU(),
            nn.Linear(config.z_dim, 9),
        )
        
    
    def forward(self, before_points, before_centroids, after_points, masks):
        '''
        before_points: (b,n,p,3)
        '''
        before_encodings = torch.stack([self.encoders[i](before_points[:,i]) for i in range(32)],dim=1)
        pos_encoding = self.pos_embed(before_centroids)
        global_points = rearrange(after_points,'b n p c -> b (n p) c')
        shape_encodings = self.shape_encoder(global_points).unsqueeze(1)
        h = before_encodings + pos_encoding
        key_masks = ~(masks.to(torch.bool))
        for layer in self.layers:
            h = layer(h, shape_encodings,key_padding_mask=key_masks)
        predicted = self.mlp(h)
        return predicted
    