export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
accelerate launch --config_file /mnt/cache/wangxiaodong/.cache/huggingface/accelerate/zero2.yaml  --mixed_precision="fp16" /mnt/cache/wangxiaodong/SDM/scripts/text_to_video/finetune_nuscene_video_textonly_v11_vis-v.py --train_batch_size=4 --gradient_accumulation_steps=2 --gradient_checkpointing --num_train_epochs 800 --learning_rate=5e-5 --max_grad_norm=1 --enable_xformers_memory_efficient_attention --lr_scheduler=cosine --lr_warmup_steps=10 --output_dir=video-v11-v-ep800-s192-5e-5 --pretrained_model_name_or_path /mnt/lustrenew/wangxiaodong/smodels/image-ep50-s192-1e-4