import sys
sys.path.append('/mnt/storage/user/wangxiaodong/DWM_work_dir/SVD/src')

import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Any
import math
import torch.nn.functional as F
from einops import rearrange, repeat

import torch.nn.init as init
from diffuser.models.attention import GatedSelfAttentionDense

from timm.models.vision_transformer import Block

def sample(logits, temperature=0.8, top_p=0.95):
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1)
    next_token = next_token.reshape(-1)
    return next_token


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class LlamaModeling(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.padding_idx = 0
        self.args = args
        self.embed_tokens = nn.Linear(args.input_dim, args.hidden_size, bias=False)
        # self.layers = nn.ModuleList([LlamaDecoderLayer(args) for _ in range(args.n_layers)]) # fewer layers for action model
        # NOTE add in preview layers
        if not self.args.add_last_layers:
            layers = [LlamaDecoderLayer(args, add_adapter=True) for _ in range(args.fusion_layers)]
            layers = layers + [LlamaDecoderLayer(args) for _ in range(args.n_layers-args.fusion_layers)]
        else:
            # NOTE add in last layers
            layers = [LlamaDecoderLayer(args) for _ in range(args.n_layers-args.fusion_layers)]
            layers = layers + [LlamaDecoderLayer(args, add_adapter=True) for _ in range(args.fusion_layers)]

        self.layers = nn.ModuleList(layers)

        self.norm = LlamaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.gradient_checkpointing = False
        self.lm_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(args.hidden_size, args.hidden_size * args.n_gram),
            nn.Linear(args.hidden_size * args.n_gram, args.input_dim * args.n_gram)
        )
        # add bos_embed
        self.bos_embed = nn.Parameter(torch.zeros(1, 1, args.hidden_size))

        self.hs_bos_embed = nn.Parameter(torch.zeros(1, 1, args.hidden_size))

        # TODO need change last blocks to be cross-attention blocks
        # p1: add multi-scale feat to hidden_states in the last_blocks
        # im_feat_in_channels = [320, 320, 640, 1280]
        # self.im_feat_attns = [ GatedSelfAttentionDense(args.hidden_size, in_chal, 8, args.hidden_size//8) for in_chal in im_feat_in_channels ]
        if not self.args.use_clip_embedding:
            if not self.args.skip_forward_visual:
                # add into last four layers
                # NOTE add adapter
                # 640, 6, 12
                self.visual_query = nn.Embedding(args.query_len, args.v_embed_dim)
                self.visual_input = nn.Linear(args.large_feat_dim, args.v_embed_dim)
                self.visual_input_norm = nn.LayerNorm(args.v_embed_dim)
                self.visual_blocks = nn.ModuleList([
                    Block(args.v_embed_dim, args.v_num_heads, args.v_mlp_ratio, qkv_bias=True)
                    for _ in range(args.v_depth)])
                self.visual_proj = nn.Linear(args.v_embed_dim, args.hidden_size)
                self.visual_proj_norm = nn.LayerNorm(args.hidden_size)

                # 1280, 3, 6
                self.visual_query_2 = nn.Embedding(args.query_len, args.v_embed_dim)
                self.visual_input_2 = nn.Linear(args.small_feat_dim, args.v_embed_dim)
                self.visual_input_norm_2 = nn.LayerNorm(args.v_embed_dim)
                self.visual_blocks_2 = nn.ModuleList([
                    Block(args.v_embed_dim, args.v_num_heads, args.v_mlp_ratio, qkv_bias=True)
                    for _ in range(args.v_depth)])
                self.visual_proj_2 = nn.Linear(args.v_embed_dim, args.hidden_size)
                self.visual_proj_norm_2 = nn.LayerNorm(args.hidden_size)

                # mid: 1280, 3, 6
                self.visual_query_3 = nn.Embedding(args.query_len, args.v_embed_dim)
                self.visual_input_3 = nn.Linear(args.small_feat_dim, args.v_embed_dim)
                self.visual_input_norm_3 = nn.LayerNorm(args.v_embed_dim)
                self.visual_blocks_3 = nn.ModuleList([
                    Block(args.v_embed_dim, args.v_num_heads, args.v_mlp_ratio, qkv_bias=True)
                    for _ in range(args.v_depth)])
                self.visual_proj_3 = nn.Linear(args.v_embed_dim, args.hidden_size)
                self.visual_proj_norm_3 = nn.LayerNorm(args.hidden_size)
            else:
                # 640, 1280, 1280
                self.visual_proj = nn.Linear(640, args.hidden_size)
                self.visual_proj_norm = nn.LayerNorm(args.hidden_size)

                self.visual_proj_1 = nn.Linear(1280, args.hidden_size)
                self.visual_proj_norm_1 = nn.LayerNorm(args.hidden_size)

                self.visual_proj_2 = nn.Linear(1280, args.hidden_size)
                self.visual_proj_norm_2 = nn.LayerNorm(args.hidden_size)
        else:
            self.clip_proj = nn.Linear(768, args.hidden_size)
            self.clip_norm = nn.LayerNorm(args.hidden_size)

            self.visual_query = nn.Embedding(args.query_len, args.v_embed_dim)
            self.visual_input = nn.Linear(args.v_embed_dim, args.v_embed_dim)
            self.visual_input_norm = nn.LayerNorm(args.v_embed_dim)
            self.visual_blocks = nn.ModuleList([
                Block(args.v_embed_dim, args.v_num_heads, args.v_mlp_ratio, qkv_bias=True)
                for _ in range(args.v_depth)])
            self.visual_proj = nn.Linear(args.v_embed_dim, args.hidden_size)
            self.visual_proj_norm = nn.LayerNorm(args.hidden_size)


    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    def forward_visual(self, action_model_input_list=None):
        large_feat = None
        small_feat = None
        mid_block_feat = None
        for t in action_model_input_list:
            if small_feat is not None:
                mid_block_feat = t
                break
            if t.shape[1] == 640:
                large_feat = t
            elif t.shape[1] == 1280:
                small_feat = t
            
        large_feat = rearrange(large_feat, 'b c h w -> b (h w) c')
        small_feat = rearrange(small_feat, 'b c h w -> b (h w) c')
        mid_block_feat = rearrange(mid_block_feat, 'b c h w -> b (h w) c')

        large_feat = self.visual_input_norm(self.visual_input(large_feat))
        small_feat = self.visual_input_norm_2(self.visual_input_2(small_feat))
        mid_block_feat = self.visual_input_norm_3(self.visual_input_3(mid_block_feat))

        visual_query = self.visual_query.weight.unsqueeze(
            0).repeat(len(large_feat), 1, 1)
        visual_query = torch.cat([visual_query, large_feat], dim=1)
        for block in self.visual_blocks:
            visual_query = block(visual_query)

        visual_query = visual_query[:, :self.args.query_len, :]
        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)

        visual_query_2 = self.visual_query_2.weight.unsqueeze(
            0).repeat(len(large_feat), 1, 1)
        visual_query_2 = torch.cat([visual_query_2, small_feat], dim=1)
        for block in self.visual_blocks_2:
            visual_query_2 = block(visual_query_2)

        visual_query_2 = visual_query_2[:, :self.args.query_len, :]
        visual_query_2 = self.visual_proj_2(visual_query_2)
        visual_query_2 = self.visual_proj_norm_2(visual_query_2)

        # for mid-block feat
        visual_query_3 = self.visual_query_3.weight.unsqueeze(
            0).repeat(len(large_feat), 1, 1)
        visual_query_3 = torch.cat([visual_query_3, mid_block_feat], dim=1)
        for block in self.visual_blocks_3:
            visual_query_3 = block(visual_query_3)

        visual_query_3 = visual_query_3[:, :self.args.query_len, :]
        visual_query_3 = self.visual_proj_3(visual_query_3)
        visual_query_3 = self.visual_proj_norm_3(visual_query_3)

        return visual_query, visual_query_2, visual_query_3


    # @torch.inference_mode()
    # NOTE we feed float input_ids as input, rather than integer
    # NOTE for unet_dwm, future video feature action_model_input_list will be feed in
    def forward(self, input_ids=None, attention_mask=None, image_context=None,
            past_key_values=None, inputs_embeds=None, use_cache=False,
            action_model_input_list=None, clip_embedding=None,
        ):

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
            # NOTE add 1, for bos token
            seq_length = seq_length + 1
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(dtype))
            # concat with bos_embed
            bsz, _, _ = inputs_embeds.shape
            bos_embed_ = repeat(self.bos_embed, 'b l d -> (exb b) l d', exb=bsz)
            bos_embed_ = bos_embed_.to(inputs_embeds.device)
            inputs_embeds = torch.cat([bos_embed_, inputs_embeds], dim=1) # b 1+l d

        # import pdb
        # pdb.set_trace()

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        # attention_mask = (attention_mask == 1)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        # decoder layers
        next_decoder_cache = () if use_cache else None

        action_hs_prediction = []

        all_hidden_states = []

        # NOTE
        if action_model_input_list is not None:
            # clip_embedding priority
            # if has clip embedding, then use it!
            if clip_embedding is not None:
                large_feat = clip_embedding
                # NOTE if use short_clip_embedding
                if self.args.use_short_clip_embedding:
                    large_feat = self.visual_input_norm(self.visual_input(large_feat))

                    visual_query = self.visual_query.weight.unsqueeze(
                        0).repeat(len(large_feat), 1, 1)
                    visual_query = torch.cat([visual_query, large_feat], dim=1)
                    for block in self.visual_blocks:
                        visual_query = block(visual_query)

                    visual_query = visual_query[:, :self.args.query_len, :]
                    visual_query = self.visual_proj(visual_query)
                    visual_query = self.visual_proj_norm(visual_query)
                    adapters = (visual_query,)
                else:
                    large_feat = self.clip_proj(large_feat)
                    large_feat = self.clip_norm(large_feat)
                    # print(f'clip feat : {large_feat.shape}') # bsz, 257, 1024
                    adapters = (large_feat,) * self.args.copy_adapters
            else:
                if not self.args.skip_forward_visual:
                    adapters = self.forward_visual(action_model_input_list)
                else:
                    large_feat = None
                    small_feat = None
                    mid_block_feat = None
                    for t in action_model_input_list:
                        if small_feat is not None:
                            mid_block_feat = t
                            break
                        if t.shape[1] == 640:
                            large_feat = t
                        elif t.shape[1] == 1280:
                            small_feat = t
                        
                    large_feat = rearrange(large_feat, 'b c h w -> b (h w) c')
                    small_feat = rearrange(small_feat, 'b c h w -> b (h w) c')
                    mid_block_feat = rearrange(mid_block_feat, 'b c h w -> b (h w) c')

                    large_feat = self.visual_proj(large_feat)
                    large_feat = self.visual_proj_norm(large_feat)

                    small_feat = self.visual_proj_1(small_feat)
                    small_feat = self.visual_proj_norm_1(small_feat)

                    mid_block_feat = self.visual_proj_2(mid_block_feat)
                    mid_block_feat = self.visual_proj_norm_2(mid_block_feat)
                    
                    # no fixed visual prompt length
                    adapters = (large_feat, small_feat, mid_block_feat)

        
        for idx, decoder_layer in enumerate(self.layers):
            # all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            #NOTE if put adapter in first three layers
            
            if self.args.add_last_layers:
                # NOTE put adapter in last three layers
                visual_map = {7:0, 6:1, 5:2, 4:3}
                if idx in [4, 5, 6, 7]:
                    idx = visual_map[idx] # replace
                else:
                    idx = -1 # not add adapter
            
            if idx in [0, 1, 2, 3]:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    adapter=adapters[idx] if idx<len(adapters) else None
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache
                )

            # # NOTE get action hs prediction
            # if idx == 3:
            #     action_hs_prediction.append(layer_outputs) # (bsz, n_words, dim)

            # if idx >= 4:
            #     # (bsz, n_words, dim), (bsz, w*h, dim)
            #     visual_feat = action_model_input_list[idx-4]
            #     visual_feat = rearrange(visual_feat, 'b c h w -> b (h w) c')
            #     layer_outputs = self.im_feat_attns[idx-4](layer_outputs, visual_feat) # get n_words of self-attention

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            hidden_states = layer_outputs[0]
            all_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        next_cache = next_decoder_cache if use_cache else None

        # lm_head_intermediate = []

        # for idx, layer in enumerate(self.lm_head):
        #     hidden_states = layer(hidden_states)
        #     if idx == 1:
        #         lm_head_intermediate.append(hidden_states)
        # logits = hidden_states
        # hidden_states_ = lm_head_intermediate[0] # return as hidden_states

        # adapt validation
        logits = self.lm_head(hidden_states)

        return {"hidden_states": hidden_states, "next_cache": next_cache, 'logits': logits, 'last_second_hidden_states': all_hidden_states[-2]}
    
    # input inputs_embeds
    @torch.inference_mode()
    def sampling(self, input_ids=None, attention_mask=None, image_context=None, past_key_values=None, inputs_embeds=None, use_cache=False):
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
            # NOTE add 1, for bos token
            seq_length = seq_length + 1
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # concat with bos_embed
            bsz, _, _ = inputs_embeds.shape

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        # attention_mask = (attention_mask == 1)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        # decoder layers
        next_decoder_cache = () if use_cache else None

        all_hidden_states = []
        for idx, decoder_layer in enumerate(self.layers):
            # all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache
            )

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            hidden_states = layer_outputs[0]
            all_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)

        next_cache = next_decoder_cache if use_cache else None
        logits = self.lm_head(hidden_states)
        return {"hidden_states": hidden_states, "next_cache": next_cache, 'logits': logits, 'last_second_hidden_states': all_hidden_states[-2]}
    

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset: q.shape[-2] + offset, :]
    sin = sin[..., offset: q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class LlamaAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            add_adapter: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

        if add_adapter:
            self.gate = torch.nn.Parameter(torch.zeros(1, num_heads, 1, 1))
    #
    # def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
    #     return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            adapter: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-2]
            kv_seq_len += offset
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, offset=offset)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # NOTE adapter
        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_v = self.v_proj(adapter).view(bsz, adapter_len, self.num_heads, self.head_dim)
            adapter_v = adapter_v.transpose(1, 2)

            if adapter_len > 1:
                adapter_k = self.k_proj(adapter).view(bsz, adapter_len, self.num_heads, self.head_dim)
                adapter_k = adapter_k.transpose(1, 2)
            


        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if adapter is not None:
            if adapter_len > 1:
                adapter_scores = torch.matmul(query_states, adapter_k.transpose(2, 3)) / math.sqrt(self.head_dim)
                adapter_scores = self.gate.tanh() * F.softmax(adapter_scores.float(), dim=-1).type_as(query_states)
                attn_output = attn_output + torch.matmul(adapter_scores, adapter_v)
            else:
                attn_output = attn_output + self.gate.tanh() * adapter_v

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, args, add_adapter=False):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=args.num_attention_heads,
            add_adapter=add_adapter
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=args.intermediate_size,
        )
        self.input_layernorm = LlamaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            adapter: torch.Tensor = None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
            adapter=adapter
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs