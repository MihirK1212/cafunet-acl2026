from ..vision_text_pipeline_common.clip_embedding import CLIPEmbeddings
from ..vision_text_pipeline_common.text_vision_fuser import TextVisionFuser
from ..vision_text_pipeline_common.block_fusion import BlockFusion
from ..vision_text_pipeline_common.hybrid_feature_enrichment import HybridFeatureEnrichment

import torch
import torch.nn as nn
from typing import Dict

from utils import config_utils, gpu_utils
from data.schemas import (
    DataItemSchema,
    PytorchModelOutputSchema,
)
import constants

config = config_utils.load_config()


DEFAULT_CLASSES = (
    constants.LABELS_CRISIS_MMD_HUMANITARIAN_CATEGORIES if config.get('crisis_mmd_like_dataset_to_use') == constants.FIELD_CRISIS_MMD_DATASET
    else constants.LABELS_TSEQD_INFORMATIVENESS_CATEGORIES
)

LABEL_CATEGORY_FIELD = (
    constants.FIELD_HUMANITARIAN_CATEGORY if config.get('crisis_mmd_like_dataset_to_use') == constants.FIELD_CRISIS_MMD_DATASET
    else constants.FIELD_INFORMATIVENESS_CATEGORY
)


class CrisisMMDVisionTextPipelineModel(nn.Module):
    def __init__(
        self,
        num_classes=len(
            DEFAULT_CLASSES
        ),
        rank: int = gpu_utils.get_device(),
    ):
        super(CrisisMMDVisionTextPipelineModel, self).__init__()

        assert (
            config[constants.FIELD_MODEL_TO_USE]
            == constants.MODEL_CRISIS_MMD_VISION_TEXT_PIPELINE
        ) or (
            config[constants.FIELD_MODEL_TO_USE]
            == constants.MODEL_CRISIS_MMD_FUZZY_ENSEMBLE
        )

        image_embedding_dim = config.get(constants.FIELD_IMAGE_EMBEDDING_DIM)
        text_embedding_dim = config.get(constants.FIELD_TEXT_EMBEDDING_DIM)

        self.clip_embeddings_model = CLIPEmbeddings()
        self.text_vision_fuser = TextVisionFuser()

        # Hybrid Feature Enrichment gates (Section 4.4.1, Eq. 9)
        self.image_enrichment = HybridFeatureEnrichment(image_embedding_dim)
        self.text_enrichment = HybridFeatureEnrichment(text_embedding_dim)

        self.block_fusion_branch = BlockFusion(
            dim_x=image_embedding_dim, dim_y=text_embedding_dim, output_dim=512
        )

        self.prediction_branch = nn.Linear(512, num_classes)

    def forward(self, data_item: DataItemSchema):
        # Frozen CLIP global embeddings: g_t, g_v (used as priors in Eq. 9)
        text_embeddings: Dict[str, torch.Tensor] = self.clip_embeddings_model(
            None, data_item.metadata["caption"]
        )["text"]
        global_text_embedding = text_embeddings.get(constants.FIELD_GLOBAL_EMBEDDING)

        image_embeddings: Dict[str, torch.Tensor] = self.clip_embeddings_model(
            data_item.metadata["image_path"], None
        )["image"]
        global_image_embedding = image_embeddings.get(constants.FIELD_GLOBAL_EMBEDDING)

        batch_size = data_item.image_tensors[
            LABEL_CATEGORY_FIELD
        ].rgb_pixels_tensor.shape[0]


        collated_topics = data_item.metadata["topics"]
        topic_strings = [collated_topics[i]["topic"][0] for i in range(len(collated_topics))]
        with torch.no_grad():
            fixed_embs = self.clip_embeddings_model(None, topic_strings)["text"].get(
                constants.FIELD_GLOBAL_EMBEDDING
            )
        topic_embs = fixed_embs.unsqueeze(0).expand(batch_size, -1, -1).detach()

        # Dual-conditioned encoders produce ĥ_v, ĥ_t  (Sections 4.2 + 4.3)
        fused_image_features, fused_text_features = self.text_vision_fuser(
            data_item, topic_embs
        )

        # Hybrid Feature Enrichment  (Section 4.4.1, Eq. 9)
        image_features = self.image_enrichment(fused_image_features, global_image_embedding)
        text_features = self.text_enrichment(fused_text_features, global_text_embedding)

        # Projected Bilinear Block Fusion  (Section 4.4.2, Eq. 10)
        feature_vector = self.block_fusion_branch(image_features, text_features)

        humanitarian_logits = self.prediction_branch(feature_vector)

        return PytorchModelOutputSchema(
            **{
                constants.FIELD_PRED_LOGITS: {
                    LABEL_CATEGORY_FIELD: humanitarian_logits,
                },
                constants.FIELD_METADATA: {
                    constants.FIELD_TEXT_EMBEDDINGS: {
                        constants.FIELD_GLOBAL_EMBEDDING: fused_text_features
                    },
                    constants.FIELD_IMAGE_EMBEDDINGS: {
                        constants.FIELD_GLOBAL_EMBEDDING: fused_image_features
                    },
                    'feature_vector': feature_vector
                },
            }
        )
