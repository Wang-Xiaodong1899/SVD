export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
python /mnt/cache/wangxiaodong/SDM/scripts/action_to_video/metrics/simple_val_action_v3_0_full.py --roll_out 1 --num_frames 8 --pretrained_model_name_or_path /mnt/lustrenew/wangxiaodong/smodels-vis/action-v30-best-5e-5/checkpoint-10000 --cfg_min 1 --cfg_max 3