from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..loaders import UNet2DConditionLoadersMixin
from ..utils import BaseOutput, logging
from .attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from .embeddings import TimestepEmbedding, Timesteps
from .modeling_utils import ModelMixin
from .unet_3d_blocks_action_wm import UNetMidBlockSpatioTemporal, get_down_block, get_up_block
from .resnet_action import Downsample2D

from einops import rearrange, repeat

from .activations import GEGLU, GELU, ApproximateGELU
from .attention_processor import Attention
from .attention import FeedForward
from .action_layers_norm import ActionCrossModule

# NOTE Real world model
# action -> vision
# video -> action


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNetSpatioTemporalConditionOutput(BaseOutput):
    sample: torch.FloatTensor = None
    action: torch.FloatTensor = None


class UNetSpatioTemporalConditionModel_Action(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types: Tuple[str] = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 512,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 20, 20), # NOTE need be (5, 10, 20, 20)
        num_frames: int = 8,
        temp_style: str = "text",
        history_len: int = 8,
        wm_action_path: str= ""
    ):
        super().__init__()

        self.sample_size = sample_size

        # input
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], True, downscale_freq_shift=0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # NOTE add time_proj and time_embedding for extra timesteps
        self.time_proj_extra = Timesteps(block_out_channels[0], True, downscale_freq_shift=0)
        self.time_embedding_extra = TimestepEmbedding(timestep_input_dim, time_embed_dim)


        self.add_time_proj = Timesteps(addition_time_embed_dim, True, downscale_freq_shift=0)
        self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        # NOTE add add_embedding projection
        # add pos_embedding
        # print(f'projection_class_embeddings_input_dim : {projection_class_embeddings_input_dim}, {time_embed_dim}') 768,1280
        self.pos_proj = Timesteps(projection_class_embeddings_input_dim, True, 0)
        self.pos_embed = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim

        #TODO: add action & context frame

        fore_channel_mult = [1, 2, 4, 4] #align with z latent code
        fore_channels = 320 # align with latent code
        self.context_block = nn.ModuleList([])
        fore_in_channel = 4 # 4 is the latent channel
        self.context_block.append(
            nn.Conv2d(fore_in_channel, fore_channels, 3, 1, 1) #64
        )
        fore_in_channel = fore_channels
        # down 3 times
        for ch_s_i, ch_s in enumerate(fore_channel_mult):
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

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i] # [320, 640, 1280, 1280]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                resnet_act_fn="silu",
                temp_style=temp_style
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockSpatioTemporal(
            block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1],
            temp_style=temp_style
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False
            
            is_same_channel = True
            if i in [1, 2]:
                is_same_channel = False
            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-5,
                resolution_idx=i,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                resnet_act_fn="silu",
                is_same_channel=is_same_channel,
                temp_style=temp_style
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-5)
        self.conv_act = nn.SiLU()

        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )

        # action branch
        self.wm_action = ActionCrossModule(cross_attention_dim=640)
        missing_keys, unexpected_keys = self.wm_action.load_state_dict(
            torch.load(wm_action_path, map_location='cpu'), strict=False
        )
        print(f'loaded wm_action from {wm_action_path}')
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = False,
        action: torch.FloatTensor = None,
        image_context: torch.FloatTensor = None,
        clip_embedding: torch.FloatTensor = None,
        history_len: int = None,
        stop_action_grad: bool = False,
        enable_train_action_encoder: bool = False,
        added_to_temb: bool = True,
        skip_video: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]

        # print(f'timesteps: {timesteps}')

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        # print(f'before emb: {emb.shape}')

        emb = emb[:, None].repeat_interleave(num_frames, dim=1) #b, f, d

        # naive version
        emb = emb # retain (b, f, d)

        # align with action_embedding
        if emb.shape[0] != batch_size:
            emb = emb.repeat(int(batch_size//emb.shape[0]), 1, 1)

        # print(f'emb shape {emb.shape}') # (1, 8, 1280)

        # NOTE asure 3 dim added_time_ids
        added_time_ids = added_time_ids.to(sample.device)

        # b his_frame+future_frame d
        history_action = added_time_ids[:, :history_len]
        pred_action_embedding = self.wm_action.encode(history_action, image_context)

        # NOTE add action_embedding
        if added_to_temb:
            emb = emb + pred_action_embedding
        
        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        emb = emb.flatten(0, 1)
        
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
        
        # take predicted action embedding as temporal attention hint
        replace_embedding = rearrange(pred_action_embedding, 'b f d -> (b f) d')
        replace_embedding = repeat(replace_embedding, 'b d -> b n d', n=1)
        
        # NOTE: action --(guide)--> video
        # remember to open the grad of action encoder
        if enable_train_action_encoder:
            clip_embedding = replace_embedding # no detach
        else:
            clip_embedding = replace_embedding.detach() # (b*f, l, d)

        # 2. pre-process
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        # TODO: add action & context frame
        context_frames = (image_context,)

        for module in self.context_block:
            image_context = module(image_context)
            context_frames =  context_frames + (image_context,)
        context_frames = context_frames[1:] # pop the extra small feature map

        #TODO: remember only add to first resblock
        down_block_res_samples = (sample,)
        for idx, downsample_block in enumerate(self.down_blocks):
            # print(f'Down Blocks idx {idx}')
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    action=action,
                    image_context=context_frames[idx],
                    clip_embedding=clip_embedding
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                    action=action,
                    image_context=context_frames[idx],
                )
            down_block_res_samples += res_samples
        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
            action=action,
            image_context=context_frames[-1], # smallest feature map
            clip_embedding=clip_embedding
        )

        video_feature = None
        # resnets lens: 2, 2, 2, 1
        # [40, 20, 10, 5]
        context_len = len(context_frames)
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            # print(f'UP Blocks idx {i}')
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :] # len=2, smallest feature map?
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    image_context = context_frames[-(i+2)],
                    clip_embedding=clip_embedding
                )
                # print(f'{i} sample shape: {sample.shape}')
                # video feature --(guide)--> action
                if i==1:
                    video_feature = sample # (dim=1280) # (8, 1280, 12, 24)
                    # print(f'got video feature: {video_feature.shape}')
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                    image_context = context_frames[-(i+2)],
                )
                # print(f'{i} sample shape: {sample.shape}')

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)

        sample = self.conv_out(sample)

        if skip_video:
            video_feature = None
        else:
            # NOTE action prediction
            # print(f'in unet_wm: {video_feature.shape}')
            if stop_action_grad:
                video_feature = video_feature.detach()
                # print('stop action gradient backprop')
        predict_action = self.wm_action.forward_with_video(pred_action_embedding, video_feature)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample, predict_action)

        return UNetSpatioTemporalConditionOutput(sample=sample, action=predict_action)
