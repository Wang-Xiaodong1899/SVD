export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
source ~/.bashrc
conda activate diff
accelerate launch --mixed_precision="fp16" /mnt/cache/wangxiaodong/SVD-Sense/scripts/text_to_image/finetune_nuscene_image.py --use_ema --train_batch_size=32 --gradient_accumulation_steps=4 --gradient_checkpointing --num_train_epochs 50 --learning_rate=1e-4 --max_grad_norm=0.1 --enable_xformers_memory_efficient_attention --lr_scheduler=cosine --lr_warmup_steps=10 --output_dir=image-ep50-ddp --pretrained_model_name_or_path /mnt/lustrenew/wangxiaodong/models/stable-diffusion-v1-4 --drop_text