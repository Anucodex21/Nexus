from .blip import BLIPCaptioner
from PIL import Image
import os

class ImageCaptioner:
    """Advanced image captioning with multiple models."""

    def __init__(self, model_type="blip"):
        self.model_type = model_type

        if model_type == "blip":
            self.captioner = BLIPCaptioner()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def generate_caption(self, image_path, style="descriptive"):
        """Generate caption with style."""
        base_caption = self.captioner.caption(image_path)

        if style == "descriptive":
            return base_caption
        elif style == "creative":
            return f"A captivating scene: {base_caption}"
        elif style == "technical":
            return f"Image analysis: {base_caption}"
        else:
            return base_caption

    def generate_detailed_caption(self, image_path):
        """Generate detailed caption with multiple aspects."""
        caption = self.captioner.caption(image_path)

        # Simulate additional details
        details = {
            "main_subject": caption,
            "mood": "neutral",
            "setting": "unknown",
            "colors": "varied"
        }

        return details

    def batch_caption(self, image_dir, output_file=None):
        """Caption all images in a directory."""
        results = {}

        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_path = os.path.join(image_dir, filename)
                caption = self.generate_caption(image_path)
                results[filename] = caption

        if output_file:
            with open(output_file, 'w') as f:
                for filename, caption in results.items():
                    f.write(f"{filename}: {caption}\n")

        return results
