import os


class LlamaConfig:
    img_size = 256
    patch_size = 16
    num_patches = 256

    # llama
    n_layers = 6 #8
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

    # ctx3_ms8
    # NOTE video model is based on this model
    # training sequence
    # his_seq_len = 3
    # history_len = 3
    # max_video_len = 5 # after action history, NOTE max_seq_len = history_len + max_video_len
    # valid_max_video_len = 36 # 
    # max_seq_len = 8 # total sequence length for transformer

    # train and validate for ctx4_ms8
    # max_seq_len = 8 # total sequence length for transformer
    # his_seq_len = 4
    # history_len = 4
    # max_video_len = 4 # after action history, NOTE max_seq_len = history_len + max_video_len
    # valid_max_video_len = 36 # 

    # ctx8_ms16
    his_seq_len = 8
    history_len = 8
    max_video_len = 8 # after action history, NOTE max_seq_len = history_len + max_video_len
    valid_max_video_len = 36 # 
    max_seq_len = 16 # total sequence length for transformer

    # test train split
    # his_seq_len = 8
    # history_len = 8
    # max_video_len = 8 # after action history, NOTE max_seq_len = history_len + max_video_len
    # valid_max_video_len = 36 # 
    # max_seq_len = 16 # total sequence length for transformer

    # ctx12_ms24
    # his_seq_len = 12
    # history_len = 12
    # max_video_len = 12 # after action history, NOTE max_seq_len = history_len + max_video_len
    # valid_max_video_len = 36 # 
    # max_seq_len = 24 # total sequence length for transformer

    # test 4 history for ctx8_ms16
    # his_seq_len = 4
    # history_len = 4
    # max_video_len = 12 # after action history, NOTE max_seq_len = history_len + max_video_len
    # valid_max_video_len = 36 # 
    # max_seq_len = 16 # total sequence length for transformer


    video_offset = 4
    prediction_scope = 4 # 4

    eval_train_sample = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
