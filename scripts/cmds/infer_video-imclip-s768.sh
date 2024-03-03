export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
python /mnt/cache/wangxiaodong/SDM/scripts/action_to_video/metrics/simple_infer_s768-v.py --pretrained_model_name_or_path /mnt/lustrenew/wangxiaodong/smodels-vis/video-v11-v-imclip-s768-5e-5/checkpoint-4000 --roll_out 5 --train_frames 8