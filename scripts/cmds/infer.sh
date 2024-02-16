export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
python /mnt/cache/wangxiaodong/SDM/scripts/text_to_video/metrics/simple_val_interpolat.py --pretrained_model_name_or_path /mnt/lustrenew/wangxiaodong/smodels-vis/interpolat-ep200-s256-L6 --num_frames 6 --train_frames 6