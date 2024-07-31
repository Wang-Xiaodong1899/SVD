import os


class LlamaConfig:
    img_size = 256
    patch_size = 16
    num_patches = 256

    # llama
    n_layers = 8
    input_dim = 2
    hidden_size = 1024
    num_attention_heads = 8
    intermediate_size = 2048
    norm_eps = 1e-5
    rms_norm_eps = 1e-6
    use_cache = False

    n_gram = 4
    
    hidden_act = "silu"

    in_chans = 4
    embed_dim = 512
    context_dim = 640

    dropout = 0.1
    cross_heads = 4
    att_dropout = 0.1
    att_heads = 4
    latent_blocks = 2

    # his action
    action_n_head = 4
    action_block_exp = 4
    action_attn_pdrop = 0.1
    action_resid_pdrop = 0.1
    action_n_layer = 2
    action_input_dim = 2

    # training sequence
    max_seq_len = 5 # total sequence length for transformer

    his_seq_len = 4
    history_len = 4
    max_video_len = 5 # after action history, NOTE max_seq_len = history_len + max_video_len
    valid_max_video_len = 5 # 

    # NOTE debug
    # training sequence
    max_seq_len = 8 # total sequence length for transformer
    his_seq_len = 4
    history_len = 4
    max_video_len = 5 # after action history, NOTE max_seq_len = history_len + max_video_len
    valid_max_video_len = 40 # 24, 40

    video_offset = 0
    

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
