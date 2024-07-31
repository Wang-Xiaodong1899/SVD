from sre_constants import OP_UNICODE_IGNORE
import torch
from torch import nn
from typing import Any, Dict, Optional
from einops import rearrange, repeat

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


class ActionAttnLayer(nn.Module):
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



class ActionCrossAttnLayer(nn.Module):
    def __init__(
        self,
        dim: int, 
        num_attention_heads: int = 10,
        cross_attention_dim: Optional[int] = None,

    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads

        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(
            dim,
            dim_out=dim,
            activation_fn="geglu",
        )

        self.norm1 = nn.LayerNorm(dim)
        # cross attn
        if cross_attention_dim is not None:
            self.attn1 = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=dim//num_attention_heads,
                cross_attention_dim=cross_attention_dim,
            )
        else:
            raise Exception("cross_attention_dim is not defined!")

        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, activation_fn="geglu")
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ):
        # input: b f d
        bsz, frames, dim = hidden_states.shape

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)
        hidden_states = self.ff_in(hidden_states)
        hidden_states = hidden_states + residual

        # print(f'context shape: {encoder_hidden_states.shape}') #work!

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states

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
        if context is not None:
            context = rearrange(context, 'b c h w -> b c (h w)')
            context = rearrange(context, 'b c n -> b n c')

        # print(f'context shape {context.shape}') (1, 72, 640)
        # print(f"action embed shape {action_embed.shape}") (1, 8, 1280)

        action_embed = self.layer1(hidden_states=action_embed, encoder_hidden_states=context)
        action_embed = self.layer2(hidden_states=action_embed, encoder_hidden_states=context)

        return action_embed


class ActionAttnEncoder(nn.Module):
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

        # print(f'context shape {context.shape}') (1, 72, 640)
        # print(f"action embed shape {action_embed.shape}") (1, 8, 1280)

        action_embed = self.layer1(hidden_states=action_embed, encoder_hidden_states=context)
        action_embed = self.layer2(hidden_states=action_embed, encoder_hidden_states=context)

        return action_embed


class ActionCoarseDecoder(nn.Module):
    def __init__(
            self, dim: int, num_attention_heads: int = 10,
            action_dim: int=2,
            cross_attention_dim: Optional[int] = None,
        ):
        super().__init__()
        self.layer1 = ActionCoarseLayer(
            dim=dim, num_attention_heads=num_attention_heads)
        self.layer2 = ActionCoarseLayer(
            dim=dim, num_attention_heads=num_attention_heads)
        self.decoder = nn.Linear(dim, action_dim)

    def forward(self, action_embed, return_feature=False):
        # NOTE decode action_embed to action without context
        action_embed = self.layer1(action_embed)
        action_embed = self.layer2(action_embed)
        action = self.decoder(action_embed)

        if return_feature:
            return (action_embed, action)

        return action


class ActionCrossAttnDecoder(ActionCoarseDecoder):
    def __init__(
            self, dim: int, num_attention_heads: int = 10,
            action_dim: int=2,
            cross_attention_dim: Optional[int] = None,
            cross_levels: int=2,
            video_feature_dim: Optional[int] = None,
        ):
        super().__init__(
            dim=dim, num_attention_heads=num_attention_heads,
            action_dim=action_dim, cross_attention_dim=cross_attention_dim
        )
        # extra layers
        self.cross_layers = torch.nn.ModuleList([
            ActionCrossAttnLayer(
                dim=dim, num_attention_heads=num_attention_heads,
                cross_attention_dim=video_feature_dim
            )
            for i in range(cross_levels)
        ])
        self.decoder_pre = ActionCoarseLayer(
            dim=dim, num_attention_heads=num_attention_heads)
    def forward(self, action_embed, encoder_hidden_states):
        bsz = action_embed.shape[0]
        if len(encoder_hidden_states.shape)==4:
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b c h w -> b c (h w)')
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b c n -> b n c') # b*f, 12x24, 1280
            # print(f'encoder_hidden_states shape: {encoder_hidden_states.shape}')
        # print(f'in ActionCrossAttnDecoder, {encoder_hidden_states.shape}') work!
        for blk, ca_blk in zip([self.layer1, self.layer2], self.cross_layers):
            action_embed = blk(action_embed)

            action_embed = rearrange(action_embed, 'b n c -> (b n) c')
            action_embed = repeat(action_embed, 'b c -> b l c', l=1) # b*f, 1, 1280
            # print(f'action embed cross shape: {action_embed.shape}')

            action_embed = ca_blk(action_embed, encoder_hidden_states) # b*f, 1, 1280
            action_embed = rearrange(action_embed, '(b f) n c -> b (f n) c', b=bsz)
            # print(f'action embed self shape: {action_embed.shape}')

        action_embed = self.decoder_pre(action_embed)

        action = self.decoder(action_embed)
        
        return action



class ActionCoarseModule(nn.Module):
    def __init__(
            self, dim: int = 1280, num_attention_heads: int = 10,
            action_dim: int=2,
            cross_attention_dim = None,
            projection_class_embeddings_input_dim: int = 512,
            time_embed_dim: int = 1280,
            addition_time_embed_dim: int = 256
        ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim

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
            action_dim=action_dim, cross_attention_dim=cross_attention_dim
        )

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
    
    def forward(self, action, image_latents):
        
        device = next(self.encoder.parameters()).device

        # action -> action embed
        bsz, num_frames = action.shape[:2]

        time_embeds = self.action_proj(action.flatten())
        time_embeds = time_embeds.reshape((bsz, num_frames, -1))
        time_embeds = time_embeds

        action_embed = self.action_embedding(time_embeds) # b f 1280

        num_frames_emb = torch.arange(num_frames, device=device)
        num_frames_emb = num_frames_emb.repeat(bsz, 1) # (b, f)
        num_frames_emb = num_frames_emb.reshape(-1) # flatten
        add_pos_emb = self.pos_proj(num_frames_emb)
        add_pos_emb = add_pos_emb.to(dtype=time_embeds.dtype)

        add_pos_emb = self.pos_embed(add_pos_emb) # (bf, d)
        add_pos_emb = add_pos_emb.reshape((bsz, num_frames, -1))

        # add position embedding
        action_embed = action_embed + add_pos_emb

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
            del context_frames
        else:
            context = None


        action_embed = self.encoder(action_embed, context)
        action_pred = self.decoder(action_embed)

        return action_pred

    def encode(self, action, image_latents, action_embed=None, context=None):
        if action is not None and image_latents is not None:
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

            context_frames = (image_latents,)
            for module in self.context_block:
                image_latents = module(image_latents)
                context_frames =  context_frames + (image_latents,)
            context_frames = context_frames[1:]

            context = context_frames[-1]
            del context_frames

            action_embed = self.encoder(action_embed, context)

        else:
            if action is None and context is not None:
                action_embed = self.encoder(action_embed, context)
            elif action is None and image_latents is not None:
                context_frames = (image_latents,)
                for module in self.context_block:
                    image_latents = module(image_latents)
                    context_frames =  context_frames + (image_latents,)
                context_frames = context_frames[1:]

                context = context_frames[-1]
                del context_frames

                action_embed = self.encoder(action_embed, context)
            else:
                raise Exception("ilegal action and context input")

        return action_embed
    

# NOTE ActionAttnModule
# normalized input action
# postion embedding
# nn.embedding as embedding
class ActionAttnModule(nn.Module):
    def __init__(
            self, dim: int = 1280, num_attention_heads: int = 10,
            action_dim: int=2,
            cross_attention_dim: Optional[int] = 1280,
            projection_class_embeddings_input_dim: int = 512,
            time_embed_dim: int = 1280,
            addition_time_embed_dim: int = 256
        ):
        super().__init__()

        steer_num = 200
        steer_offset = 80

        speed_num = 700
        speed_offset = 0

        # embedder for action
        self.steer_embedding = nn.Embedding(steer_num, time_embed_dim)
        self.speed_embedding = nn.Embedding(speed_num, time_embed_dim)

        max_frame_pos = 16
        # position for action
        self.pos_embedding = nn.Embedding(max_frame_pos, time_embed_dim)
        
        self.encoder = ActionCoarseEncoder(
            dim=dim, num_attention_heads=num_attention_heads,
            cross_attention_dim=cross_attention_dim
        )
        self.decoder = ActionCoarseDecoder(
            dim=dim, num_attention_heads=num_attention_heads,
            action_dim=action_dim, cross_attention_dim=cross_attention_dim
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
    
    def forward(self, action, image_latents, return_feature=False):
        # action -> action embed
        bsz, num_frames = action.shape[:2]

        # time_embeds = self.action_proj(action.flatten())
        # time_embeds = time_embeds.reshape((bsz, num_frames, -1))
        # time_embeds = time_embeds.to(image_latents.dtype)
        steer = action[:, :, 0]
        speed = action[:, :, 1]

        steer_embed = self.steer_embedding(steer) # b f d
        speed_embed = self.speed_embedding(speed) # b f d

        action_embed = steer_embed + speed_embed

        action_embed = action_embed.to(image_latents.dtype) # b f 1280

        frame_position_ids = torch.arange(num_frames, dtype=torch.long, device=image_latents.device)
        frame_position_ids = frame_position_ids.unsqueeze(0).expand(bsz, -1)

        add_pos_emb = self.pos_embedding(frame_position_ids) # b f d

        # add position embedding
        action_embed = action_embed + add_pos_emb

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
        del context_frames


        action_embed = self.encoder(action_embed, context)
        action_pred = self.decoder(action_embed, return_feature)

        return action_pred

    def encode(self, action, image_latents, action_embed=None, context=None):
        if action is not None and image_latents is not None:
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

            context_frames = (image_latents,)
            for module in self.context_block:
                image_latents = module(image_latents)
                context_frames =  context_frames + (image_latents,)
            context_frames = context_frames[1:]

            context = context_frames[-1]
            del context_frames

            action_embed = self.encoder(action_embed, context)

        else:
            if action is None and context is not None:
                action_embed = self.encoder(action_embed, context)
            elif action is None and image_latents is not None:
                context_frames = (image_latents,)
                for module in self.context_block:
                    image_latents = module(image_latents)
                    context_frames =  context_frames + (image_latents,)
                context_frames = context_frames[1:]

                context = context_frames[-1]
                del context_frames

                action_embed = self.encoder(action_embed, context)
            else:
                raise Exception("ilegal action and context input")

        return action_embed
 

class ActionCrossModule(nn.Module):
    def __init__(
            self, dim: int = 1280, num_attention_heads: int = 10,
            action_dim: int=2,
            cross_attention_dim: Optional[int] = 1280,
            projection_class_embeddings_input_dim: int = 512,
            time_embed_dim: int = 1280,
            addition_time_embed_dim: int = 256,
            video_feature_dim: Optional[int] = 1280,
        ):
        super().__init__()

        # embedder for action
        self.action_proj = Timesteps(addition_time_embed_dim, True, downscale_freq_shift=0)
        self.action_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        # position for action
        self.pos_proj = Timesteps(projection_class_embeddings_input_dim, True, 0)
        self.pos_embed = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        self.encoder = ActionCoarseEncoder(
            dim=dim, num_attention_heads=num_attention_heads,
            cross_attention_dim=cross_attention_dim
        )
        self.decoder = ActionCrossAttnDecoder(
            dim=dim, num_attention_heads=num_attention_heads,
            action_dim=action_dim, cross_attention_dim=cross_attention_dim,
            video_feature_dim=video_feature_dim
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
        del context_frames


        action_embed = self.encoder(action_embed, context)
        action_pred = self.decoder(action_embed)

        return action_pred

    def encode(self, action, image_latents, action_embed=None, context=None):
        if action is not None and image_latents is not None:
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

            context_frames = (image_latents,)
            for module in self.context_block:
                image_latents = module(image_latents)
                context_frames =  context_frames + (image_latents,)
            context_frames = context_frames[1:]

            context = context_frames[-1]
            del context_frames

            action_embed = self.encoder(action_embed, context)
        else:
            raise Exception("ilegal action and context input")

        return action_embed
    
    def forward_with_video(self, action_embed, video_features):
        # print(f'in action_layers_norm: {video_features.shape}') work!
        output = self.decoder(action_embed, video_features)

        return output
    
