import torch
import torch.nn as nn
import clip
from torchvision import transforms
from PIL import Image, ImageFile
from data.schemas import DataItemSchema, PytorchModelOutputSchema
import constants
from utils import config_utils, caption_utils, gpu_utils
from copy import deepcopy

config = config_utils.load_config()

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CLIPEmbeddings(nn.Module):
    def __init__(
        self,
        clip_model_name="ViT-B/32",
    ) -> None:
        super(CLIPEmbeddings, self).__init__()

        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load(
            clip_model_name, device=gpu_utils.get_device()
        )
        self.device = gpu_utils.get_device()

    def forward(
        self, image_paths: list[str], captions: list[str]
    ):
        return {
            **self._get_image_embeddings(image_paths),
            **self._get_text_embeddings(captions)
        }
        
    def _get_image_embeddings(self, image_paths: list[str]):
        if image_paths is None:
            return {}
        images = [
            self.clip_preprocess(Image.open(img_path).convert("RGB"))
            for img_path in image_paths
        ]
        images = torch.stack(images).to(self.device)

        # Image global embeddings
        with torch.no_grad():
            image_global_embedding = self.clip_model.encode_image(images).to(torch.float32)

        # Image patch embeddings
        with torch.no_grad():
            image_features = self.clip_model.visual.conv1(images.half())
        batch_size, channels, height, width = image_features.shape
        image_patch_embeddings = image_features.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channels
        )
        image_patch_embeddings = image_patch_embeddings.to(torch.float32)

        return {
            "image": {
                constants.FIELD_GLOBAL_EMBEDDING: image_global_embedding,
                constants.FIELD_WORD_LEVEL_EMBEDDINGS: image_patch_embeddings
            },
        }
    
    def _get_text_embeddings(self, captions: list[str]):
        if captions is None:
            return {}
        
        # Text global embedding
        text_global_embedding = self._get_global_embeddings_for_captions(captions)

        # Text group embeddings
        processed_tensors = []
        group_size = 3
        for caption in captions:
            padded_caption = caption_utils.pad_caption(
                caption, config.get(constants.FIELD_MAX_LEN_TEXT)
            )
            caption_groups = CLIPEmbeddings.group_caption_words(
                padded_caption, group_size
            )
            caption_tensor = self._get_global_embeddings_for_captions(
                caption_groups
            )
            processed_tensors.append(caption_tensor)
        text_group_embeddings = torch.stack(
            processed_tensors
        )  # Shape: (num_captions, num_groups, embed_dim)

        text_global_embedding = text_global_embedding.to(torch.float32)
        text_group_embeddings = text_group_embeddings.to(torch.float32)

        return {
            "text": {
                constants.FIELD_GLOBAL_EMBEDDING: text_global_embedding,  # [batch_size, 512]
                constants.FIELD_WORD_LEVEL_EMBEDDINGS: text_group_embeddings,  # [batch_size, max_len_tokens, 512]
            },
        }

    def _get_global_embeddings_for_captions(self, captions: list[str]):
        captions = deepcopy(captions)
        text_tokens = clip.tokenize(captions, truncate=True).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
        return text_embeddings

    @classmethod
    def group_caption_words(cls, caption: str, group_size: int) -> list[str]:
        words = caption.split()
        return [
            " ".join(words[i : i + group_size])
            for i in range(0, len(words), group_size)
        ]