export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
python /mnt/cache/wangxiaodong/SDM/scripts/action_to_video/metrics/Eval_fvd_dir_sort.py --i3d_device "cuda" --tgt_dir /mnt/lustrenew/wangxiaodong/data/nuscene/FVD-first-36-action/action-v30-base-5e-5-checkpoint-10000-3-3-step25 --eval_frames 16 --skip_gt 0