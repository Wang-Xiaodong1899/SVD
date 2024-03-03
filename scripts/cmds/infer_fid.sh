export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
# srun -p vc7 -n1 --gres=gpu:1 --ntasks-per-node=1
python -m pytorch_fid /mnt/lustrenew/wangxiaodong/data/nuscene_val/gt /mnt/lustrenew/wangxiaodong/data/nuscene_val/image-v-s768-1e-4-checkpoint-8000