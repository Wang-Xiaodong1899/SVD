export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
accelerate launch --config_file /mnt/cache/wangxiaodong/.cache/huggingface/accelerate/zero2.yaml  --mixed_precision="fp16" /mnt/cache/wangxiaodong/SDM/scripts/joint_output/finetune_action_s192.py --train_batch_size=6 --gradient_accumulation_steps=2 --gradient_checkpointing --num_train_epochs 50000 --checkpointing_steps 3000 --learning_rate=5e-5 --max_grad_norm=1 --enable_xformers_memory_efficient_attention --lr_scheduler=cosine --lr_warmup_steps=10 --output_dir=AV_MLP-5e-5 --temp_style="image" --img_style="pooler_output" --pretrained_model_name_or_path /mnt/lustrenew/wangxiaodong/smodels/image-ep50-s192-1e-4