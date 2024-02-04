export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
python /mnt/cache/wangxiaodong/SDM/scripts/text_to_video/metrics/simple_val_firstframe.py --pretrained_model_name_or_path /mnt/lustrenew/wangxiaodong/smodels/video-v01-ep200-s192-L12 --num_frames 12 --train_frames 12