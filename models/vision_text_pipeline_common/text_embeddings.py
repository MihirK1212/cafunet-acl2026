import torch
import torch.nn as nn
from transformers import RobertaModel
from typing import Dict

import constants
from data.schemas import TokenizedTextInputsSchema

class BertTextEmbeddingsModel(nn.Module):
    def __init__(self):
        super(BertTextEmbeddingsModel, self).__init__()
        self.plm = RobertaModel.from_pretrained("roberta-base")

        self.plm_num_hidden_layers = self.plm.config.num_hidden_layers
        self.sentence_weights = torch.nn.parameter.Parameter(
            torch.Tensor(1, self.plm_num_hidden_layers)
        )

    def get_global_vector_embedding(self, plm_output):
        vg =  plm_output.pooler_output       

        # hidden_state = plm_output[0]
        # vg = hidden_state[:, 0]

        # print('vg shape:', vg.shape)
        
        return vg

    def get_weighted_word_level_embeddings(self, plm_output):
        layer_outputs = torch.stack(
            plm_output["hidden_states"][-self.plm_num_hidden_layers :], dim=2
        ).permute(0, 2, 1, 3)
        reshaped_weights = self.sentence_weights.view(
            1, self.plm_num_hidden_layers, 1, 1
        )
        weighted_sum = torch.sum(layer_outputs * reshaped_weights, dim=1)
        return weighted_sum

    def forward(self, tokenized_text_inputs: TokenizedTextInputsSchema) -> Dict[str, torch.Tensor]:
        plm_output = self.plm(
            input_ids=tokenized_text_inputs.ids,
            attention_mask=tokenized_text_inputs.attention_mask,
            token_type_ids=tokenized_text_inputs.token_type_ids,
            output_hidden_states=True,
        )
        vg = self.get_global_vector_embedding(plm_output)
        H = self.get_weighted_word_level_embeddings(plm_output)
        return {
            constants.FIELD_GLOBAL_EMBEDDING: vg,
            constants.FIELD_WORD_LEVEL_EMBEDDINGS: H
        }
