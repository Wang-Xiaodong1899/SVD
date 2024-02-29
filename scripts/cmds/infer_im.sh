export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
python /mnt/cache/wangxiaodong/SDM/scripts/text_to_video/ge_img_nuscene.py --base_model '/mnt/lustrenew/wangxiaodong/smodels/image-e-s192-1e-4' --checkpoint 20000