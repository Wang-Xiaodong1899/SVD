export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
MASTER_PORT=5560
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

echo "MASTER_ADDR"=$MASTER_ADDR
echo "NNODES"=$NNODES
echo "NODE_RANK"=$NODE_RANK

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
accelerate launch --multi_gpu  --machine_rank $SLURM_PROCID --num_machines 2 --num_processes 16 --main_process_port $MASTER_PORT --main_process_ip "$MASTER_ADDR" --config_file /mnt/cache/wangxiaodong/.cache/huggingface/accelerate/ddp-multi.yaml --mixed_precision="fp16" /mnt/cache/wangxiaodong/SDM/scripts/text_to_video/finetune_nuscene_video_textonly_v11-v-imclip.py --train_batch_size=1 --gradient_accumulation_steps=2 --gradient_checkpointing --num_train_epochs 10 --learning_rate=5e-5 --max_grad_norm=1 --enable_xformers_memory_efficient_attention --lr_scheduler=cosine --lr_warmup_steps=10 --output_dir=video-v11-v-imclip-ep10-s192-5e-5-fixnorm --temp_style="image" --img_style="pooler_output" --pretrained_model_name_or_path /mnt/lustrenew/wangxiaodong/smodels/image-ep50-s192-1e-4