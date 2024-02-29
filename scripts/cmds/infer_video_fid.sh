export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
python /mnt/cache/wangxiaodong/SDM/scripts/text_to_video/metrics/fid.py --tgt_dir /mnt/lustrenew/wangxiaodong/data/nuscene/FVD-first-15-rec/video-v11-v-imclip-s192-5e-5-fixnorm-checkpoint-40000 --eval_frames 38