export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
python /mnt/cache/wangxiaodong/SDM/scripts/text_to_video/metrics/Eval_fvd_dir_sort.py --i3d_device "cuda:0" --tgt_dir /mnt/lustrenew/wangxiaodong/data/nuscene/FVD-first-15-rec/im192-video-v11-ep100-s192 --eval_frames 8