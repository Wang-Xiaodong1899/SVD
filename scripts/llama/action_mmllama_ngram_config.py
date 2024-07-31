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

    # adapter
    v_embed_dim=768
    v_depth=2
    v_num_heads=16
    v_mlp_ratio=4.0
    query_len=10
    large_feat_dim = 640
    small_feat_dim = 1280
    fusion_layers = 3 # add mid-block feat # default

    # train ID: action-mmllama-1e-4-v8_bs4_h8_ms16_4gram_last_layers_sequence-clip-pooler_copy4/last.pth
    # Continue train: action-mmllama-1e-4-v8_bs4_h8_ms16_4gram_last_layers_sequence-clip-pooler_copy4_load_capybara-35
    # fusion_layers = 4
    # random_timestep = True
    # skip_forward_visual = False
    # add_last_layers = True
    # use_clip_embedding = True
    # use_short_clip_embedding = False
    # copy_adapters = 4
    # history_video = 8 # used history video
    # use_sequence_imgs = True # NOTE Attention!!! TODO

    # train ID: action-mmllama-1e-4-v8_bs4_h8_ms16_4gram_sequence_clip_pooler_copy1_last_1_layer
    # fusion_layers = 1
    # random_timestep = True
    # skip_forward_visual = False
    # add_last_layers = True
    # use_clip_embedding = True
    # use_short_clip_embedding = False
    # copy_adapters = 1
    # history_video = 8 # used history video
    # use_sequence_imgs = True # NOTE Attention!!! TODO

    # train ID: action-mmllama-1e-4-v8_bs4_h8_ms16_4gram_sequence-4_clip_pooler_copy1_last_1_layer
    fusion_layers = 1
    random_timestep = True
    skip_forward_visual = False
    add_last_layers = True
    use_clip_embedding = True
    use_short_clip_embedding = False
    copy_adapters = 1
    history_video = 4 # used history video
    use_sequence_imgs = True # NOTE Attention!!! TODO

    # train ID: action-mmllama-1e-4-v8_bs4_h8_ms16_4gram_sequence-4_clip_pooler_copy4_last_4_layer
    # fusion_layers = 4
    # random_timestep = True
    # skip_forward_visual = False
    # add_last_layers = True
    # use_clip_embedding = True
    # use_short_clip_embedding = False
    # copy_adapters = 4
    # history_video = 4 # used history video
    # use_sequence_imgs = True # NOTE Attention!!! TODO



    # test action-mmllama-1e-4-v8_bs4_h8_ms16_4gram_clip_short_first_layers_ep400
    # random_timestep = True
    # skip_forward_visual = False
    # add_last_layers = False

    # use_clip_embedding = True
    # use_short_clip_embedding = True
    # copy_adapters = 3

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
    # ctx3_ms7
    # max_seq_len = 7 # total sequence length for transformer
    # his_seq_len = 3
    # history_len = 3
    # max_video_len = 4 # after action history, NOTE max_seq_len = history_len + max_video_len
    # valid_max_video_len = 36 # 

    # ctx8_ms16
    # max_seq_len = 16 # total sequence length for transformer
    # his_seq_len = 8
    # history_len = 8
    # max_video_len = 8 # after action history, NOTE max_seq_len = history_len + max_video_len
    # valid_max_video_len = 16 # 

    # NOTE evaluation
    max_seq_len = 16 # total sequence length for transformer
    his_seq_len = 8
    history_len = 8
    max_video_len = 8 # after action history, NOTE max_seq_len = history_len + max_video_len
    valid_max_video_len = 36 # 

    # test multi-frame clip embedding, enable use_sequence_imgs
    

    video_offset = 4
    prediction_scope = 1

    # test eval on train samples to validate overfitting
    eval_train_sample = False

    # unet
    noise_timestep = 100
    

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
