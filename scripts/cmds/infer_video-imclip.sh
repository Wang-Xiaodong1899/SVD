export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
python /mnt/cache/wangxiaodong/SDM/scripts/text_to_video/metrics/simple_val_firstframe-v-imclip.py --pretrained_model_name_or_path /mnt/lustrenew/wangxiaodong/smodels-vis/video-v11-v-imclip-ep100-s192-5e-5-fixnorm --roll_out 5 --train_frames 8