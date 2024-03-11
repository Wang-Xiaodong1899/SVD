export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
python /mnt/cache/wangxiaodong/SDM/scripts/action_to_video/metrics/fid.py --tgt_dir /mnt/lustrenew/wangxiaodong/data/nuscene/FVD-first-8-action/action-v30-base-5e-5-checkpoint-10000-3-3-step50 --eval_frames 8