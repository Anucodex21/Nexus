import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class BLIPCaptioner:
    """Generate captions for images using BLIP."""

    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def caption(self, image_path, max_length=50):
        """Generate caption for an image."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=max_length)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption

    def caption_batch(self, image_paths, max_length=50):
        """Generate captions for multiple images."""
        captions = []
        for path in image_paths:
            caption = self.caption(path, max_length)
            captions.append(caption)
        return captions

    def conditional_caption(self, image_path, text_condition, max_length=50):
        """Generate caption conditioned on text."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, text_condition, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=max_length)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption
