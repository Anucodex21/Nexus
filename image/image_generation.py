from .stable_diffusion import StableDiffusionGenerator
from PIL import Image
import os

class ImageGenerator:
    """Unified image generation interface."""

    def __init__(self, backend="stable_diffusion"):
        self.backend = backend

        if backend == "stable_diffusion":
            self.generator = StableDiffusionGenerator()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def generate(self, prompt, **kwargs):
        """Generate image from prompt."""
        return self.generator.generate(prompt, **kwargs)

    def generate_from_template(self, template, variables):
        """Generate using a prompt template."""
        prompt = template.format(**variables)
        return self.generate(prompt)

    def generate_batch(self, prompts, **kwargs):
        """Generate multiple images."""
        images = []
        for prompt in prompts:
            image = self.generate(prompt, **kwargs)
            images.append(image)
        return images

    def create_variations(self, image_path, num_variations=4):
        """Create variations of an existing image."""
        # This would use img2img pipeline
        # Simplified implementation
        image = Image.open(image_path)
        variations = []

        for i in range(num_variations):
            # Apply random transformations
            variation = image.copy()
            variations.append(variation)

        return variations

    def upscale(self, image_path, scale_factor=2):
        """Upscale an image."""
        image = Image.open(image_path)
        new_size = (image.width * scale_factor, image.height * scale_factor)
        upscaled = image.resize(new_size, Image.LANCZOS)
        return upscaled
