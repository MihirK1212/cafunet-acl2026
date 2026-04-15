import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
from transformers import AutoTokenizer

from utils import config_utils, gpu_utils
from data.schemas import (
    DataItemSchema,
    PytorchModelOutputSchema,
    TokenizedTextInputsSchema,
)
import constants

from .fuzz_feature_extractor import FuzzyFeatureExtractor

config = config_utils.load_config()

VOCAB_SIZE = config.get(constants.FIELD_VOCAB_SIZE)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, ffe: FuzzyFeatureExtractor | None = None):
        # CGC at last layer (Eq. 8): x = x ⊙ (1 + G(x))
        if ffe is not None:
            x_batch = x.permute(1, 0, 2)
            gating_score = ffe(x_batch)
            x = (x_batch * (1 + gating_score)).permute(1, 0, 2)

        x = (
            x
            + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        )
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, ffe: FuzzyFeatureExtractor | None = None):
        for block in self.resblocks:
            x = block(x, ffe)
        return x


class VisionEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        text_embedding_dim: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, width, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.class_embedding = nn.Parameter(torch.randn(width) * width**-0.5)
        self.positional_embedding = nn.Parameter(
            torch.randn((image_size // patch_size) ** 2 + 1, width) * width**-0.5
        )

        # Topic Alignment Projector (TAP): F_TAP: R^{DT} -> R^{DV}  (Eq. 2)
        self.topic_modality_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, width),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, output_dim) * width**-0.5)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        # CGC gating function (Section 4.3)
        self.ffe = FuzzyFeatureExtractor(
            mu_params={"mu": 0.0, "sigma": 1.0},
            sigma_params={"alpha": 1.0, "beta": 0.0},
            trapezoidal_params={"a": -1.0, "b": 0.0, "c": 1.0, "d": 2.0},
            weights={"w_mu": 0.3, "w_sigma": 0.4, "w_T": 0.3, "b": 0.0},
        )

    def forward(self, image_paths: list[str], topic_embeddings: torch.Tensor):
        images = (
            torch.stack(
                [self.transform(Image.open(p).convert("RGB")) for p in image_paths]
            )
            .to(gpu_utils.get_device())
            .to(torch.float32)
        )

        x = self.conv1(images)
        x = x.flatten(2).permute(2, 0, 1)                          # (NV, batch, width)
        cls_exp = self.class_embedding.unsqueeze(0).expand(x.shape[1], -1).unsqueeze(0)
        x = torch.cat([cls_exp, x], dim=0)                         # (NV+1, batch, width)
        x = x + self.positional_embedding.unsqueeze(1)              # add pos embeddings

        # TAP: project topics into visual space  (Eq. 2)
        image_topic_embeddings = self.topic_modality_projection(topic_embeddings)  # (batch, Nt, width)
        num_topics = image_topic_embeddings.shape[1]

        # Prepend projected topics to patch sequence  (Eq. 3): E'_V = [T̃ ; E_V]
        x = x.permute(1, 0, 2)                                     # (batch, NV+1, width)
        x = torch.cat((image_topic_embeddings, x), dim=1)          # (batch, Nt+NV+1, width)

        x = x.permute(1, 0, 2)                                     # (Nt+NV+1, batch, width)
        x = self.ln_pre(x)
        # CGC (Eq. 8b) is applied at the last transformer layer
        x = self.transformer(x, self.ffe)

        x = self.ln_post(x[num_topics])                             # (batch, width)
        return x @ self.proj


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(
            torch.randn(context_length, width) * width**-0.5
        )

        self.transformer = Transformer(width, layers, heads)
        self.ln_final = LayerNorm(width)
        self.text_projection = nn.Parameter(
            torch.randn(width, output_dim) * width**-0.5
        )

        # CGC gating function (Section 4.3)
        self.ffe = FuzzyFeatureExtractor(
            mu_params={"mu": 0.0, "sigma": 1.0},
            sigma_params={"alpha": 1.0, "beta": 0.0},
            trapezoidal_params={"a": -1.0, "b": 0.0, "c": 1.0, "d": 2.0},
            weights={"w_mu": 0.3, "w_sigma": 0.4, "w_T": 0.3, "b": 0.0},
        )

    def forward(self, x: torch.Tensor, topic_embeddings: torch.Tensor):
        x = self.token_embedding(x) + self.positional_embedding.unsqueeze(0)  # (batch, NT, width)
        # (batch, Nt, width)
        num_topics = topic_embeddings.shape[1]

        # Prepend topics directly to token sequence  (Eq. 1): E'_T = [T ; E_T]
        x = torch.cat((topic_embeddings, x), dim=1)                            # (batch, Nt+NT, width)

        x = x.permute(1, 0, 2)                                                 # (Nt+NT, batch, width)
        # CGC (Eq. 8a) is applied at the last transformer layer
        x = self.transformer(x, self.ffe).permute(1, 0, 2)                     # (batch, Nt+NT, width)

        x = self.ln_final(x[:, num_topics, :])                                 # (batch, width)
        return x @ self.text_projection


class TextVisionFuser(nn.Module):
    def __init__(
        self,
    ):
        super(TextVisionFuser, self).__init__()

        self.vision_encoder = VisionEncoder(
            image_size=224,
            patch_size=16,
            width=768,
            layers=12,
            heads=8,
            output_dim=512,
            text_embedding_dim=512,
        )
        self.text_encoder = TextEncoder(
            vocab_size=VOCAB_SIZE,
            context_length=config.get(constants.FIELD_TRANSFORMER_TEXT_CONTEXT_LENGTH),
            width=512,
            layers=12,
            heads=8,
            output_dim=512,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def forward(self, data_item: DataItemSchema, topic_embeddings: torch.Tensor):
        image_paths = data_item.metadata["image_path"]
        captions = data_item.metadata["caption"]

        tokenized_texts = []
        for caption in captions:
            tokenized_texts.append(
                self.tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    max_length=config.get('transformer_text_context_length'),
                    return_tensors="pt",
                )["input_ids"]
                .to(torch.long)
                .to(gpu_utils.get_device())
            )
        tokenized_texts = (
            torch.squeeze(torch.stack(tokenized_texts), dim=1)
            .to(torch.long)
            .to(gpu_utils.get_device())
        )
        tokenized_texts[tokenized_texts >= VOCAB_SIZE] = VOCAB_SIZE - 1
        assert tokenized_texts.shape == (len(captions), config.get('transformer_text_context_length')), tokenized_texts.shape

        image_features = self.vision_encoder(image_paths, topic_embeddings)
        text_features = self.text_encoder(tokenized_texts, topic_embeddings)

        return image_features, text_features
