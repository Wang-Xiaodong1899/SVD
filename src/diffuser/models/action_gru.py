

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from einops import rearrange, repeat
import numpy as np
from functools import partial

from .attention import FeedForward, Attention
from .resnet_action import Downsample2D


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)

class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int, 
        num_attention_heads: int = 10,
        cross_attention_dim: Optional[int] = None,

    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.ff_in = FeedForward(
            dim,
            dim_out=dim,
            activation_fn="geglu",
        )
        # self attn
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=dim//num_attention_heads,
        )
        # cross attn
        if cross_attention_dim is not None:
            self.attn2 = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=dim//num_attention_heads,
                cross_attention_dim=cross_attention_dim,
            )
        else:
            self.attn2 = None

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, activation_fn="geglu")
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ):
        # input: b f d
        bsz, frames, dim = hidden_states.shape
        hidden_states = self.ff_in(hidden_states)

        attn_output = self.attn1(hidden_states, encoder_hidden_states=None)
        hidden_states = hidden_states + attn_output

        if self.attn2 is not None:
            attn_output = self.attn2(hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = hidden_states + attn_output
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ff(hidden_states)

        return hidden_states

class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Action_GRU(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1,
        cross_attention_dim: int = 640, num_attention_heads: int = 16
        ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim

        self.gru_1 = GRUCell(size, size, norm, act, update_bias) # self gru
        self.gru_2 = GRUCell(size, size, norm, act, update_bias) # action gru
        self.gru_1.apply(weight_init)
        self.gru_2.apply(weight_init)

        self.action_embedding = nn.Linear(inp_size, size, bias=False)
        self.action_embedding.apply(weight_init)

        # cross attn
        if cross_attention_dim is not None:
            self.attention = AttentionBlock(size, 
                num_attention_heads,
                cross_attention_dim,
            )
            self.attention.apply(weight_init)
        else:
            self.attention = None
        
        if cross_attention_dim is not None:
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

        self.decoder = nn.Linear(size, inp_size)

        self.context_block.apply(weight_init)
        self.decoder.apply(weight_init)

    
    def forward(self, action, image_latents, tgt_len):
        '''
            action: b f 2
            im_latent: b c h w
            tgt_len: target action len
        '''
        bsz, f, action_size = action.shape

        action_emb = self.action_embedding(action) # b f d


        # NOTE im_latent
        if self.cross_attention_dim is not None:

            context_frames = (image_latents,)
            for module in self.context_block:
                image_latents = module(image_latents)
                # print(f'image_latents shape: {image_latents.shape}')
                context_frames =  context_frames + (image_latents,)
            context_frames = context_frames[1:]

            # （1, 320, 24, 48)
            #  (1, 320, 12, 24)
            #  (1, 640, 6, 12)

            context = context_frames[-1]
            context = rearrange(context, 'b c h w -> b (h w) c')

            del context_frames
        else:
            context = None


        prev = torch.zeros_like(action_emb[:, 0])
        gru_1_outputs = []

        for i in range(f):
            _, ht = self.gru_1(action_emb[:, i], [prev])
            ht = ht[0]
            gru_1_outputs.append(ht)
            prev = ht

        action_hs = gru_1_outputs[-1] # b d


        prev = action_hs
        gru_2_outputs = []

        for i in range(tgt_len):
            x = prev
            if self.attention is not None:
                x = x[:, None]
                attn_output = self.attention(x, encoder_hidden_states=context)
                x = x + attn_output
                x = x.squeeze()

            _, ht = self.gru_2(x, [prev])
            ht = ht[0]
            gru_2_outputs.append(ht[:, None])
            prev = ht
        
        output_feat = torch.cat(gru_2_outputs, dim=1) # b f d

        outputs = self.decoder(output_feat) # b f 2

        
        # b tgt_len d
        return output_feat, outputs


class Action_PureGRU(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1,
        ):
        super().__init__()

        self.gru_1 = GRUCell(size, size, norm, act, update_bias) # self gru
        self.gru_1.apply(weight_init)

        self.action_embedding = nn.Linear(inp_size, size, bias=False)
        self.action_embedding.apply(weight_init)
    
        self.decoder = nn.Linear(size, inp_size)
        self.decoder.apply(weight_init)
    
    def forward(self, action):
        '''
            action: b f 2
            im_latent: b c h w
            tgt_len: target action len
        '''
        bsz, f, action_size = action.shape

        action_emb = self.action_embedding(action) # b f d

        prev = torch.zeros_like(action_emb[:, 0])
        gru_1_outputs = []

        for i in range(f):
            _, ht = self.gru_1(action_emb[:, i], [prev])
            ht = ht[0]
            gru_1_outputs.append(ht)
            prev = ht

        action_hs = gru_1_outputs[-1] # b d

        outputs = self.decoder(action_hs) # b 2

        outputs = outputs[:, None]
        action_hs = action_hs[:, None]

        # b tgt_len d
        return action_hs, outputs



class Improved_Action_PureGRU(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1,
        ):
        super().__init__()

        self.size = size
        self.inp_size = inp_size

        self.gru_1 = GRUCell(size, size, norm, act, update_bias) # self gru
        self.gru_1.apply(weight_init)

        # improve action embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(inp_size, size),
            nn.ReLU(inplace=True),
            nn.Linear(size, size),
        )

        self.action_embedding.apply(weight_init)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.decoder_norm = norm_layer(self.size)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.size, self.size * 2, bias=True),
            nn.Dropout(0.3),
            nn.Linear(self.size * 2, self.size),
            nn.ReLU(inplace=True),
            nn.Linear(self.size, inp_size)
        )

        self.decoder.apply(weight_init)
    
    def forward(self, action):
        '''
            action: b f 2
            im_latent: b c h w
            tgt_len: target action len
        '''
        bsz, f, action_size = action.shape

        action_emb = self.action_embedding(action) # b f d

        prev = torch.zeros_like(action_emb[:, 0])
        gru_1_outputs = []

        for i in range(f):
            _, ht = self.gru_1(action_emb[:, i], [prev])
            ht = ht[0]
            gru_1_outputs.append(ht)
            prev = ht

        action_hs = gru_1_outputs[-1] # b d

        action_hs_norm = self.decoder_norm(action_hs)

        outputs = self.decoder(action_hs_norm) # b 2

        outputs = outputs[:, None]
        action_hs = action_hs[:, None]

        # b tgt_len d
        return action_hs, outputs



# test
# action_size = 2
# model = Action_GRU(action_size, 1024)
# print('model initilized!')
# bsz = 2
# f = 8


# action = torch.randn((bsz, f, action_size))

# im = torch.randn((bsz, 4, 24, 48))

# output_1, output_2 = model(action, im, tgt_len=8)

# print(output_1.shape)
# print(output_2.shape)

# 6M gru_1 cell
        



