
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from functools import partial
from itertools import repeat
import collections.abc
import math
import torch.nn.functional as F
from torch.distributions import normal

from .attention import FeedForward, Attention
from .resnet_action import Downsample2D

import numpy as np

import torch

from einops import rearrange, repeat

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed




# =============================================================================
# Processing blocks
# X-attention input
#   Q/z_input         -> (#latent_embs, batch_size, embed_dim)
#   K/V/x             -> (#events, batch_size, embed_dim)
#   key_padding_mask  -> (batch_size, #event)
# output -> (#latent_embs, batch_size, embed_dim)
# =============================================================================
class AttentionBlock(nn.Module):  # PerceiverAttentionBlock
    def __init__(self, opt_dim, heads, dropout, att_dropout, kdim=None, vdim=None, **args):
        super(AttentionBlock, self).__init__()

        norm_x_dim = kdim if kdim else opt_dim

        self.layer_norm_x = nn.LayerNorm([norm_x_dim])
        self.layer_norm_1 = nn.LayerNorm([opt_dim])
        self.layer_norm_att = nn.LayerNorm([opt_dim])

        self.attention = nn.MultiheadAttention(
            opt_dim,  # embed_dim
            heads,  # num_heads
            dropout=att_dropout,
            bias=True,
            add_bias_kv=True,
            kdim=kdim,
            vdim=vdim
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(opt_dim, opt_dim)
        self.layer_norm_2 = nn.LayerNorm([opt_dim])
        self.linear2 = nn.Linear(opt_dim, opt_dim)
        self.linear3 = nn.Linear(opt_dim, opt_dim)

    def forward(self, x, z_input, mask=None, q_mask=None, **args):
        x = self.layer_norm_x(x)
        z = self.layer_norm_1(z_input)

        z_att, _ = self.attention(z, x, x, key_padding_mask=mask, attn_mask=q_mask)  # Q, K, V

        z_att = z_att + z_input
        z = self.layer_norm_att(z_att)

        z = self.dropout(z)
        z = self.linear1(z)
        z = torch.nn.GELU()(z)

        z = self.layer_norm_2(z)
        z = self.linear2(z)
        z = torch.nn.GELU()(z)
        z = self.dropout(z)
        z = self.linear3(z)

        return z + z_att


class TransformerBlock(nn.Module):
    def __init__(self, opt_dim, latent_blocks, dropout, att_dropout, heads, **args):
        super(TransformerBlock, self).__init__()

        self.latent_attentions = nn.ModuleList([
            AttentionBlock(opt_dim=opt_dim, heads=heads, dropout=dropout, att_dropout=att_dropout) for _ in
            range(latent_blocks)
        ])

    def forward(self, x_input, z, mask=None, q_mask=None, **args):
        # self-attention
        for latent_attention in self.latent_attentions:
            z = latent_attention(x_input, z, q_mask=q_mask)
        return z

class CrossTransformerBlock(nn.Module):
    def __init__(self, opt_dim, latent_blocks, dropout, att_dropout, heads, kdim=None, vdim=None, **args):
        super(CrossTransformerBlock, self).__init__()

        self.latent_attentions = nn.ModuleList([
            AttentionBlock(opt_dim=opt_dim, heads=heads, dropout=dropout, att_dropout=att_dropout, kdim=kdim, vdim=vdim) for _ in
            range(latent_blocks)
        ])

    def forward(self, x_input, z, mask=None, q_mask=None, **args):
        # self-attention
        for latent_attention in self.latent_attentions:
            z = latent_attention(x_input, z, q_mask=q_mask)
        return z


class CrossAttentionBlock(nn.Module):
    def __init__(self, opt_dim, dropout, att_dropout, cross_heads, kdim=None, vdim=None, **args):
        super(CrossAttentionBlock, self).__init__()

        self.cross_attention = AttentionBlock(opt_dim=opt_dim, heads=cross_heads, dropout=dropout,
                                              att_dropout=att_dropout, kdim=kdim, vdim=vdim)

    def forward(self, x_input, z, mask=None, q_mask=None, **args):
        # cross-attention
        z = self.cross_attention(x_input, z, mask=mask, q_mask=q_mask)
        return z


class ActionModel(nn.Module):
    def __init__(self, config):
        super(ActionModel, self).__init__()
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.in_chans = config.in_chans
        self.embed_dim = config.embed_dim
        self.att_heads = config.att_heads
        self.cross_heads = config.cross_heads
        self.latent_blocks = config.latent_blocks
        self.dropout = config.dropout
        self.att_dropout = config.att_dropout

        self.num_patches = config.num_patches
        self.context_dim = config.context_dim

        self.with_visual = self.context_dim is not None

        if self.with_visual:
            self.pos_embed_rgb = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.context_dim),
                                            requires_grad=False)  # fixed sin-cos embedding

            self.action_attention = CrossTransformerBlock(opt_dim=self.embed_dim,
                                                    latent_blocks=self.latent_blocks,
                                                    dropout=self.dropout,
                                                    att_dropout=self.att_dropout,
                                                    heads=self.att_heads,
                                                    kdim=self.context_dim,
                                                    vdim=self.context_dim
                                                    )

            self.self_attention_head = TransformerBlock(opt_dim=self.context_dim,
                                                        latent_blocks=self.latent_blocks,
                                                        dropout=self.dropout,
                                                        att_dropout=self.att_dropout,
                                                        heads=self.att_heads)
            
        self.action_input_dim = config.action_input_dim
        self.action_n_head = config.action_n_head
        self.action_block_exp = config.action_block_exp
        self.action_attn_pdrop = config.action_attn_pdrop
        self.action_resid_pdrop = config.action_resid_pdrop
        self.action_n_layer = config.action_n_layer

        self.transformer_encoder = nn.Sequential(*[Block(self.embed_dim, self.action_n_head,
                                                         self.action_block_exp, self.action_attn_pdrop,
                                                         self.action_resid_pdrop)
                                                   for _ in range(self.action_n_layer)])

        self.pos_emb_angle = nn.Parameter(torch.zeros(1, config.his_seq_len, self.embed_dim))
        self.drop = nn.Dropout(0.1)
        self.angle_emd = nn.Sequential(
            nn.Linear(self.action_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.embed_dim),
        )

        if self.with_visual:
            fore_channel_mult = [1, 2, 4, 4] #align with z latent code
            fore_channels = 320 # align with latent code
            self.context_block = nn.ModuleList([])
            fore_in_channel = 4 # 4 is the latent channel
            self.context_block.append(
                nn.Conv2d(fore_in_channel, fore_channels, 3, 1, 1) #64
            )
            fore_in_channel = fore_channels
            # down 3 times
            for ch_s_i, ch_s in enumerate(fore_channel_mult[:2]):
                layers_t = []
                layers_t.extend([
                    nn.GroupNorm(num_channels=fore_in_channel, num_groups=32, eps=1e-5),
                    nn.SiLU(),
                    nn.Conv2d(fore_in_channel, ch_s * fore_channels, 3, 1, 1),]
                )
                fore_in_channel = ch_s * fore_channels
                if ch_s_i != len(fore_channel_mult)-1: # not convsample
                    layers_t.append(
                        Downsample2D(
                            fore_in_channel, use_conv=True, out_channels=fore_in_channel, name="op",
                        )
                    )
                self.context_block.append(nn.Sequential(*layers_t))

        # decoder
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.decoder_norm = norm_layer(self.embed_dim)
        self.decoder_pred = nn.Linear(self.embed_dim, self.patch_size ** 2 * self.in_chans, bias=True)
        self.decoder = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.patch_size ** 2 * self.in_chans, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.action_input_dim)
        )
        #
        self.initialize_weights()
    def initialize_weights(self):

        if self.with_visual:
            # initialize (and freeze) pos_embed by sin-cos embedding
            pos_embed_rgb = get_2d_sincos_pos_embed(self.pos_embed_rgb.shape[-1], int(self.num_patches ** .5),
                                                    cls_token=True)
            self.pos_embed_rgb.data.copy_(torch.from_numpy(pos_embed_rgb).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, action, image_latents):
        '''
            action: b f 2
            im_latent: b c h w
            tgt_len: target action len
            suppose the im_latent is h*h, square
        '''
        bsz, f, action_size = action.shape

        action_emb = self.angle_emd(action) # b f d

        # add pos_emb
        action_emb = self.pos_emb_angle + action_emb
        action_emb = self.drop(action_emb)

        # pass transformer layers
        action_emb = self.transformer_encoder(action_emb) # b f d

        # no visual solution
        if not self.with_visual:
            emd = action_emb
        else:
            # prepare image latents
            context_frames = (image_latents,)
            for module in self.context_block:
                image_latents = module(image_latents)
                # print(f'image_latents shape: {image_latents.shape}')
                context_frames =  context_frames + (image_latents,)
            context_frames = context_frames[1:]
            
            # if 512x512 ->/8 /4 ->
            # (1, 640, 16, 16)

            context = context_frames[-1]
            context = rearrange(context, 'b c h w -> b (h w) c') # b 256 dim

            del context_frames

            # add img pos_emb
            context = context + self.pos_embed_rgb[:, 1:, :]
            context = context.permute(1, 0, 2) # l b d

            context = self.self_attention_head(context, context)

            # cross-attention
            action_emb = action_emb.permute(1, 0, 2)
            emd = self.action_attention(context, action_emb) # (k, v), q
            emd = emd.permute(1, 0, 2) # b f d

        emd = self.decoder_norm(emd)
        emd = self.decoder_pred(emd)
        emd_ = emd[:, 0]

        output = self.decoder(emd_)[:, None]

        return output





        