export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
python /mnt/cache/wangxiaodong/SDM/scripts/action_to_video/ge_img_nuscene.py --base_model '/mnt/lustrenew/wangxiaodong/models/stable-diffusion-2-1' --unet_path '/mnt/lustrenew/wangxiaodong/smodels/image-v-s768-1e-4/checkpoint-8000'