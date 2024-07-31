from sre_constants import OP_UNICODE_IGNORE
import torch
from torch import nn
from typing import Any, Dict, Optional
from einops import rearrange

from .attention import FeedForward, Attention
from .resnet_action import Downsample2D
from .embeddings import TimestepEmbedding, Timesteps
        

class ActionCoarseLayer(nn.Module):
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
        num_frames: Optional[int] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        residual: Optional[torch.FloatTensor] = None,
    ):
        # input: b f d
        bsz, frames, dim = hidden_states.shape
        hidden_states = self.ff_in(hidden_states)

        # residual only use in decoder last layer
        if residual is not None:
            hidden_states = torch.cat([residual, hidden_states], dim=1) # b, 2f, d

        attn_output = self.attn1(hidden_states, encoder_hidden_states=None)
        hidden_states = hidden_states + attn_output

        if self.attn2 is not None:
            attn_output = self.attn2(hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = hidden_states + attn_output
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ff(hidden_states)

        return hidden_states
    

class ActionCoarseEncoder(nn.Module):
    def __init__(
            self, dim: int, num_attention_heads: int = 10,
            cross_attention_dim: Optional[int] = None,
        ):
        super().__init__()
        self.layer1 = ActionCoarseLayer(
            dim=dim, num_attention_heads=num_attention_heads,
            cross_attention_dim=cross_attention_dim)
        self.layer2 = ActionCoarseLayer(
            dim=dim, num_attention_heads=num_attention_heads,
            cross_attention_dim=cross_attention_dim)
        
    def forward(self, action_embed, context):
        # action_embed: b f d
        # context: b c h w
        context = rearrange(context, 'b c h w -> b c (h w)')
        context = rearrange(context, 'b c n -> b n c')

        action_embed = self.layer1(hidden_states=action_embed, encoder_hidden_states=context)
        action_embed = self.layer2(hidden_states=action_embed, encoder_hidden_states=context)

        return action_embed


class ActionCoarseDecoder(nn.Module):
    def __init__(
            self, dim: int, num_attention_heads: int = 10,
            action_dim: int=2,
            cross_attention_dim: Optional[int] = None,
            decode_with_residual: Optional[bool] = False,
        ):
        super().__init__()
        self.layer1 = ActionCoarseLayer(
            dim=dim, num_attention_heads=num_attention_heads)
        self.layer2 = ActionCoarseLayer(
            dim=dim, num_attention_heads=num_attention_heads)
        self.decoder = nn.Linear(dim, action_dim)
        self.decode_with_residual = decode_with_residual

        # add pos if decode with residual
        if decode_with_residual:
            self.pos_proj = Timesteps(256, True, 0)
            self.pos_embed = TimestepEmbedding(256, dim)


    def forward(self, action_embed, residual=None):
        # NOTE decode action_embed to action without context
        bsz, f, d = action_embed.shape
        
        action_embed = self.layer1(action_embed)
        if self.decode_with_residual:
            bsz, rf, d = residual.shape
            num_frames_emb = torch.arange(f+rf, device=action_embed.device)
            num_frames_emb = num_frames_emb.repeat(bsz, 1) # (b, f)
            num_frames_emb = num_frames_emb.reshape(-1) # flatten
            add_pos_emb = self.pos_proj(num_frames_emb)
            add_pos_emb = add_pos_emb.to(dtype=action_embed.dtype)
            add_pos_emb = self.pos_embed(add_pos_emb) # (bf, d)
            add_pos_emb = add_pos_emb.reshape((bsz, f+rf, -1))

            action_embed = self.layer2(action_embed, residual=residual)
        else:
            action_embed = self.layer2(action_embed)
        action = self.decoder(action_embed)[:, -f:, :]

        return action


class ActionCoarseModule(nn.Module):
    def __init__(
            self, dim: int = 1280, num_attention_heads: int = 10,
            action_dim: int=2,
            cross_attention_dim: Optional[int] = 1280,
            projection_class_embeddings_input_dim: int = 512,
            time_embed_dim: int = 1280,
            addition_time_embed_dim: int = 256,
            decode_with_residual: bool = False,
        ):
        super().__init__()

        # embedder for action
        self.action_proj = Timesteps(addition_time_embed_dim, True, downscale_freq_shift=0)
        self.action_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        # position for action
        self.pos_proj = Timesteps(projection_class_embeddings_input_dim, True, 0)
        self.pos_embed = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        # add_modules = [nn.Linear(time_embed_dim * num_frames, time_embed_dim)]
        # for _ in range(1):
        #     add_modules.append(nn.GELU())
        #     add_modules.append(nn.Linear(time_embed_dim, num_frames * cross_attention_dim))
        # self.add_embedding_projector = nn.Sequential(*add_modules)
        
        self.encoder = ActionCoarseEncoder(
            dim=dim, num_attention_heads=num_attention_heads,
            cross_attention_dim=cross_attention_dim
        )
        self.decoder = ActionCoarseDecoder(
            dim=dim, num_attention_heads=num_attention_heads,
            action_dim=action_dim, cross_attention_dim=cross_attention_dim,
            decode_with_residual=decode_with_residual
        )

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
        
        self.decode_with_residual = decode_with_residual
    
    def forward(self, action, image_latents):
        # action -> action embed
        bsz, num_frames = action.shape[:2]

        time_embeds = self.action_proj(action.flatten())
        time_embeds = time_embeds.reshape((bsz, num_frames, -1))
        time_embeds = time_embeds.to(image_latents.dtype)

        action_embed = self.action_embedding(time_embeds) # b f 1280

        num_frames_emb = torch.arange(num_frames, device=image_latents.device)
        num_frames_emb = num_frames_emb.repeat(bsz, 1) # (b, f)
        num_frames_emb = num_frames_emb.reshape(-1) # flatten
        add_pos_emb = self.pos_proj(num_frames_emb)
        add_pos_emb = add_pos_emb.to(dtype=image_latents.dtype)

        add_pos_emb = self.pos_embed(add_pos_emb) # (bf, d)
        add_pos_emb = add_pos_emb.reshape((bsz, num_frames, -1))

        # add position embedding
        action_embed = action_embed + add_pos_emb

        residual = None
        if self.decode_with_residual:
            residual = action_embed
        
        context_frames = (image_latents,)
        for module in self.context_block:
            image_latents = module(image_latents)
            context_frames =  context_frames + (image_latents,)
        context_frames = context_frames[1:]

        context = context_frames[-1]
        del context_frames

        # print(f'image context shape: {context.shape}')
        # import pdb
        # pdb.set_trace()

        action_embed = self.encoder(action_embed, context)
        action_pred = self.decoder(action_embed, residual)

        return action_pred

    def encode(self, action_embed, context):
        action_embed = self.encoder(action_embed, context)

        return action_embed
    
