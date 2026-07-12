import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import os

class StableDiffusionGenerator:
    """Generate images using Stable Diffusion."""

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.pipe = self.pipe.to(self.device)

        # Enable memory efficient attention if available
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()

    def generate(self, prompt, num_inference_steps=50, guidance_scale=7.5,
                 height=512, width=512, num_images=1, seed=None):
        """Generate image from text prompt."""
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        images = self.pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_images_per_prompt=num_images,
            generator=generator
        ).images

        return images[0] if num_images == 1 else images

    def generate_variations(self, prompt, num_variations=4):
        """Generate multiple variations."""
        return self.generate(prompt, num_images=num_variations)

    def save_image(self, image, path):
        """Save generated image."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        image.save(path)
        return path
