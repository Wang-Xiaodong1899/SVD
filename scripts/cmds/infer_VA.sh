export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo $MASTER_ADDR
python /mnt/cache/wangxiaodong/SDM/scripts/joint_output/metrics/simple_val_full_same_t.py --roll_out 1 --num_frames 8 --pretrained_model_name_or_path /mnt/lustrenew/wangxiaodong/smodels-vis/VA_MLP-5e-5-same-T/checkpoint-1000 --cfg_min 1 --cfg_max 1