Backup from Neurips 2024 rebuttal
## prepare Environment
```
# cuda12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install diffusers==0.25.0 transformers transforms3d einops nuscenes-devkit safetensors

```

## Start run
```
python scripts/text_to_video/simple_infer_s448-v_nb.py
```