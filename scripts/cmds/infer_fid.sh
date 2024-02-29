export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
python -m pytorch_fid /mnt/lustrenew/wangxiaodong/data/nuscene_val/gt /mnt/lustrenew/wangxiaodong/data/nuscene_val/image-e-s192-1e-4-20000 