import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from consistencydecoder import ConsistencyDecoder, save_image, load_image

device="cuda:0"

# encode with stable diffusion vae
# pipe = StableDiffusionPipeline.from_pretrained(
#     "/volsparse1/wxd/smodels/stable-diffusion-v1-4", torch_dtype=torch.float16, device="cuda:0"
# )
vae = AutoencoderKL.from_pretrained(
                '/volsparse1/wxd/smodels/stable-diffusion-v1-4', subfolder="vae",
    )
vae.eval()
vae.to(device)

decoder_consistency = ConsistencyDecoder(device="cuda:0") # Model size: 2.49 GB

image = load_image("/root/SVD/scripts/text_to_video/process/cd.png", size=(800, 448), center_crop=False)
latent = vae.encode(image.cuda()).latent_dist.mean

# decode with gan
sample_gan = vae.decode(latent).sample.detach()
save_image(sample_gan, "/root/SVD/scripts/text_to_video/process/gan.png")

# decode with vae
sample_consistency = decoder_consistency(latent)
save_image(sample_consistency, "/root/SVD/scripts/text_to_video/process/con.png")