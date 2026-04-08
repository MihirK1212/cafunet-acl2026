import torch
from typing import Dict

from data.schemas import PytorchModelOutputSchema
from utils import config_utils, gpu_utils
import constants

config = config_utils.load_config()


class VisionTextPipelineLoss(torch.nn.Module):
    def __init__(self, rank: int = gpu_utils.get_device(), class_weights=None) -> None:
        super(VisionTextPipelineLoss, self).__init__()
        self.classification_loss_function = torch.nn.CrossEntropyLoss(
            weight = class_weights.to(rank)
            # **(
            #     dict(weight=class_weights.to(rank))
            #     if (class_weights is not None)
            #     else {}
            # )
        )
        self.temperature = config.get(constants.FIELD_CL_LOSS_TEMPERATURE)
        self.rank = rank

    def _get_cross_entropy_loss(
        self, model_output: PytorchModelOutputSchema, target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pred_logits = model_output.pred_logits
        assert pred_logits.keys() == target.keys()
        loss = 0
        for k in pred_logits.keys():
            loss += self.classification_loss_function(pred_logits[k], target[k])
        return loss

    def _compute_cl_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        Computes the contrastive loss (CL) for a batch of embeddings using cosine similarity
        and a cross-entropy loss function.

        This method is designed for contrastive learning where paired embeddings (`z1` and `z2`)
        are processed to maximize similarity within pairs (positive pairs) and minimize similarity
        between non-paired embeddings (negative pairs).

        Args:
            z1 (torch.Tensor): A tensor of shape `(batch_size, embedding_dim)` representing the first set of embeddings.
            z2 (torch.Tensor): A tensor of shape `(batch_size, embedding_dim)` representing the second set of embeddings.

        Returns:
            torch.Tensor: The computed contrastive loss value as a scalar tensor.

        Details:
        - **Positive Pairs:**
            - For a sample `i` in the original batch:
                - `z1[i]` and `z2[i]` form a positive pair.
            - After forming the combined batch `z` (alternating between elements of `z1` and `z2`),
            the embeddings at even indices are paired with the embeddings at the next odd indices.
            - For example, `z[0]` (from `z1[0]`) and `z[1]` (from `z2[0]`) are a positive pair,
            as are `z[2]` (from `z1[1]`) and `z[3]` (from `z2[1]`), and so on.
        - **Negative Pairs:**
            - All other pairs of embeddings in the combined batch `z` are treated as negatives.
            - For example, `z[0]` (from `z1[0]`) and `z[3]` (from `z2[1]`) form a negative pair,
            as do `z[1]` (from `z2[0]`) and `z[4]` (from `z1[2]`), etc.

        Process:
        - A combined batch `z` is created by interleaving `z1` and `z2`, resulting in a tensor of shape `(2 * batch_size, embedding_dim)`.
        - Cosine similarity scores are computed for all possible pairs in `z`, resulting in a similarity matrix of shape `(combined_batch_size, combined_batch_size)`.
        - The diagonal elements of the similarity matrix (self-similarities) are set to `-inf` to exclude them from consideration.
        - Target indices are constructed such that for each embedding in `z`, its paired embedding (positive pair) is specified as the correct target.
            - Example: For `z[0]` (from `z1[0]`), the target is `1` (index of `z2[0]`).
        - Cross-entropy loss is computed using the scaled cosine similarity scores and the target indices to enforce high similarity for positive pairs and low similarity for negatives.
        - The loss is scaled down by a factor of 10.0.

        Notes:
        - `self.temperature` is expected to be a scalar value controlling the temperature scaling in the contrastive loss.
        - Ensure that `self.temperature` is a positive float to avoid division errors.
        """
        z = []
        for b in range(z1.shape[0]):
            z.append(z1[b])
            z.append(z2[b])
        z = torch.stack(z, dim=0)
        z.to(self.rank)
        combined_batch_size = z.shape[0]
        xcs = torch.nn.functional.cosine_similarity(
            z[None, :, :], z[:, None, :], dim=-1
        )
        xcs[torch.eye(combined_batch_size).bool()] = float("-inf")
        target = torch.arange(combined_batch_size, device=self.rank)
        target[0::2] += 1
        target[1::2] -= 1
        xcs.to(self.rank)
        target.to(self.rank)
        cl_loss = torch.nn.functional.cross_entropy(
            xcs / self.temperature, target, reduction="mean"
        )
        cl_loss /= 10.0
        return cl_loss

    def forward(
        self, model_output: PytorchModelOutputSchema, target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        cross_entropy_loss = self._get_cross_entropy_loss(model_output, target)
        cl_loss = 0

        text_global_embedding = model_output.metadata[constants.FIELD_TEXT_EMBEDDINGS][
            constants.FIELD_GLOBAL_EMBEDDING
        ]
        # text_token_embeddings = model_output.metadata[constants.FIELD_TEXT_EMBEDDINGS][
        #     constants.FIELD_WORD_LEVEL_EMBEDDINGS
        # ]

        image_global_embedding = model_output.metadata[
            constants.FIELD_IMAGE_EMBEDDINGS
        ][constants.FIELD_GLOBAL_EMBEDDING]
        # image_token_embeddings = model_output.metadata[
        #     constants.FIELD_IMAGE_EMBEDDINGS
        # ][constants.FIELD_WORD_LEVEL_EMBEDDINGS]

        # batch_size = text_global_embedding.shape[0]

        cl_loss += self._compute_cl_loss(text_global_embedding, image_global_embedding)
        # for b in range(batch_size):
        #     cl_loss += self._compute_cl_loss(
        #         text_token_embeddings[b], image_token_embeddings[b]
        #     )

        return cross_entropy_loss + cl_loss
