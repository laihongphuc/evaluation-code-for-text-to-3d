from typing import Union, List

import torch
import torch.nn as nn
from jaxtyping import Float
from PIL import Image
from transformers import CLIPTokenizer, CLIPModel, CLIPImageProcessor


class ClipFeatureExtractor(nn.Module):
    def __init__(self,
                 model_name="openai/clip-vit-base-patch32",
                 pretrained=True):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        # text and image preprocessor
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
    
    @torch.no_grad()
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        if not isinstance(text, list):
            text = [text]
        device = self.parameters().__next__().device
        text_inputs = self.tokenizer(text=text, padding=True, truncation=True, return_tensors="pt")["input_ids"].to(device)
        text_features = self.model.get_text_features(text_inputs)
        return text_features

    @torch.no_grad()
    def encode_image(self, image: Float[torch.Tensor, "N C H W"]) -> torch.Tensor:
        return self.model.get_image_features(image)
    
    def image_preprocessor(self, image: Image) -> Float[torch.Tensor, "C H W"]:
        return self.image_processor(image, return_tensors="pt")['pixel_values'][0]