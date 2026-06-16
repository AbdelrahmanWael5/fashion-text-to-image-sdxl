import torch
from diffusers import StableDiffusionXLPipeline


class ClothingGenerator:

    def __init__(
        self,
        model_path,
        checkpoint_path,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):

        self.device = device

        print("Device =", device)
        
        dtype = (
            torch.float16
            if device == "cuda"
            else torch.float32
        )
        
        print("Loading SDXL...")
        
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)

        print("Pipeline loaded")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        print("Checkpoint loaded")

        self.pipe.unet.load_state_dict(
            checkpoint["unet_state_dict"]
        )

        self.pipe.unet.eval()

        print("UNET loaded")

        


    @torch.no_grad()
    def generate_image(
        self,
        prompt,
        negative_prompt="low quality, blurry, watermark, deformed",
        seed=42,
        guidance_scale=7.5,
        num_inference_steps=30,
        height=512,
        width=384,
    ):

        generator = torch.Generator(
            device=self.device
        ).manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

        return image