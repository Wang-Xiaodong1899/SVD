import os


class GlobalConfig:
    img_size = 256
    patch_size = 16
    num_patches = 256

    in_chans = 4
    embed_dim = 512
    context_dim = None

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


    his_seq_len = 8
    

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
